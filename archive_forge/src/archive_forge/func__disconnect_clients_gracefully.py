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
def _disconnect_clients_gracefully(self, status=CLOSE_STATUS_NORMAL, reason=DEFAULT_CLOSE_REASON):
    """
        Terminate clients gracefully without shutting down the server
        """
    for client in self.clients:
        client['handler'].send_close(status, reason)
    self._terminate_client_handlers()