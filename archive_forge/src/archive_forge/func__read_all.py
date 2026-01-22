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