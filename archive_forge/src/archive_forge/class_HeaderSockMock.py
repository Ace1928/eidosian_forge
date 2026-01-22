import os
import os.path
import socket
import ssl
import unittest
import websocket
import websocket as ws
from websocket._http import (
class HeaderSockMock(SockMock):

    def __init__(self, fname):
        SockMock.__init__(self)
        path = os.path.join(os.path.dirname(__file__), fname)
        with open(path, 'rb') as f:
            self.add_packet(f.read())