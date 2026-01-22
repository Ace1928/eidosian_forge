import logging
import socket
import sys
import threading
from socket import AF_INET, SOCK_STREAM
from ssl import CERT_REQUIRED, SSLContext, SSLError
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TLSError
from ncclient.transport.session import Session
from ncclient.transport.parser import DefaultXMLParser
def _transport_read(self):
    return self._socket.recv(BUF_SIZE)