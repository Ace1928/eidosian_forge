import os
import socket
import struct
import sys
import threading
import time
import tempfile
import stat
from logging import DEBUG
from select import select
from paramiko.common import io_sleep, byte_chr
from paramiko.ssh_exception import SSHException, AuthenticationException
from paramiko.message import Message
from paramiko.pkey import PKey, UnknownKeyType
from paramiko.util import asbytes, get_logger
class AgentSSH:

    def __init__(self):
        self._conn = None
        self._keys = ()

    def get_keys(self):
        """
        Return the list of keys available through the SSH agent, if any.  If
        no SSH agent was running (or it couldn't be contacted), an empty list
        will be returned.

        This method performs no IO, just returns the list of keys retrieved
        when the connection was made.

        :return:
            a tuple of `.AgentKey` objects representing keys available on the
            SSH agent
        """
        return self._keys

    def _connect(self, conn):
        self._conn = conn
        ptype, result = self._send_message(cSSH2_AGENTC_REQUEST_IDENTITIES)
        if ptype != SSH2_AGENT_IDENTITIES_ANSWER:
            raise SSHException('could not get keys from ssh-agent')
        keys = []
        for i in range(result.get_int()):
            keys.append(AgentKey(agent=self, blob=result.get_binary(), comment=result.get_text()))
        self._keys = tuple(keys)

    def _close(self):
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._keys = ()

    def _send_message(self, msg):
        msg = asbytes(msg)
        self._conn.send(struct.pack('>I', len(msg)) + msg)
        data = self._read_all(4)
        msg = Message(self._read_all(struct.unpack('>I', data)[0]))
        return (ord(msg.get_byte()), msg)

    def _read_all(self, wanted):
        result = self._conn.recv(wanted)
        while len(result) < wanted:
            if len(result) == 0:
                raise SSHException('lost ssh-agent')
            extra = self._conn.recv(wanted - len(result))
            if len(extra) == 0:
                raise SSHException('lost ssh-agent')
            result += extra
        return result