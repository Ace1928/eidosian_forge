import sys
import struct
import ssl
from base64 import b64encode
from hashlib import sha1
import logging
from socket import error as SocketError
import errno
import threading
from socketserver import ThreadingMixIn, TCPServer, StreamRequestHandler
from websocket_server.thread import WebsocketServerThread
def _client_left_(self, handler):
    client = self.handler_to_client(handler)
    self.client_left(client, self)
    if client in self.clients:
        self.clients.remove(client)