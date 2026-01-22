import errno
import fcntl
import os
from oslo_log import log as logging
import select
import signal
import socket
import ssl
import struct
import sys
import termios
import time
import tty
from urllib import parse as urlparse
import websocket
from zunclient.common.apiclient import exceptions as acexceptions
from zunclient.common.websocketclient import exceptions
class WebSocketClient(BaseClient):

    def __init__(self, zunclient, url, id, escape='~', close_wait=0.5):
        super(WebSocketClient, self).__init__(zunclient, url, id, escape, close_wait)

    def connect(self):
        url = self.url
        LOG.debug('connecting to: %s', url)
        try:
            self.ws = websocket.create_connection(url, skip_utf8_validation=True, origin=self._compute_origin_header(url), sslopt={'cert_reqs': ssl.CERT_REQUIRED, 'ca_certs': self.get_system_ca_file()}, subprotocols=['binary', 'base64'])
            print('connected to %s, press Enter to continue' % self.id)
            print('type %s. to disconnect' % self.escape)
        except socket.error as e:
            raise exceptions.ConnectionFailed(e)
        except websocket.WebSocketConnectionClosedException as e:
            raise exceptions.ConnectionFailed(e)
        except websocket.WebSocketBadStatusException as e:
            raise exceptions.ConnectionFailed(e)

    def _compute_origin_header(self, url):
        origin = urlparse.urlparse(url)
        if origin.scheme == 'wss':
            return 'https://%s:%s' % (origin.hostname, origin.port)
        else:
            return 'http://%s:%s' % (origin.hostname, origin.port)

    def fileno(self):
        return self.ws.fileno()

    def send(self, data):
        self.ws.send_binary(data)

    def recv(self):
        return self.ws.recv()

    @staticmethod
    def get_system_ca_file():
        """Return path to system default CA file."""
        ca_path = ['/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/ca-bundle.pem', '/etc/ssl/cert.pem']
        for ca in ca_path:
            if os.path.exists(ca):
                return ca
        return None