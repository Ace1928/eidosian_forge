from binascii import hexlify
import getpass
import inspect
import os
import socket
import warnings
from errno import ECONNREFUSED, EHOSTUNREACH
from paramiko.agent import Agent
from paramiko.common import DEBUG
from paramiko.config import SSH_PORT
from paramiko.dsskey import DSSKey
from paramiko.ecdsakey import ECDSAKey
from paramiko.ed25519key import Ed25519Key
from paramiko.hostkeys import HostKeys
from paramiko.rsakey import RSAKey
from paramiko.ssh_exception import (
from paramiko.transport import Transport
from paramiko.util import ClosingContextManager
def _families_and_addresses(self, hostname, port):
    """
        Yield pairs of address families and addresses to try for connecting.

        :param str hostname: the server to connect to
        :param int port: the server port to connect to
        :returns: Yields an iterable of ``(family, address)`` tuples
        """
    guess = True
    addrinfos = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    for family, socktype, proto, canonname, sockaddr in addrinfos:
        if socktype == socket.SOCK_STREAM:
            yield (family, sockaddr)
            guess = False
    if guess:
        for family, _, _, _, sockaddr in addrinfos:
            yield (family, sockaddr)