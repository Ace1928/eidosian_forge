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
class AgentRemoteProxy(AgentProxyThread):
    """
    Class to be used when wanting to ask a remote SSH Agent
    """

    def __init__(self, agent, chan):
        AgentProxyThread.__init__(self, agent)
        self.__chan = chan

    def get_connection(self):
        return (self.__chan, None)