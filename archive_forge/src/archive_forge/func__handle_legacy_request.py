import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
def _handle_legacy_request(self, environ):
    if 'eventlet.input' in environ:
        sock = environ['eventlet.input'].get_socket()
    elif 'gunicorn.socket' in environ:
        sock = environ['gunicorn.socket']
    else:
        raise Exception('No eventlet.input or gunicorn.socket present in environ.')
    if 'HTTP_SEC_WEBSOCKET_KEY1' in environ:
        self.protocol_version = 76
        if 'HTTP_SEC_WEBSOCKET_KEY2' not in environ:
            raise BadRequest()
    else:
        self.protocol_version = 75
    if self.protocol_version == 76:
        key1 = self._extract_number(environ['HTTP_SEC_WEBSOCKET_KEY1'])
        key2 = self._extract_number(environ['HTTP_SEC_WEBSOCKET_KEY2'])
        environ['wsgi.input'].content_length = 8
        key3 = environ['wsgi.input'].read(8)
        key = struct.pack('>II', key1, key2) + key3
        response = md5(key).digest()
    scheme = 'ws'
    if environ.get('wsgi.url_scheme') == 'https':
        scheme = 'wss'
    location = '%s://%s%s%s' % (scheme, environ.get('HTTP_HOST'), environ.get('SCRIPT_NAME'), environ.get('PATH_INFO'))
    qs = environ.get('QUERY_STRING')
    if qs is not None:
        location += '?' + qs
    if self.protocol_version == 75:
        handshake_reply = b'HTTP/1.1 101 Web Socket Protocol Handshake\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\nWebSocket-Origin: ' + environ.get('HTTP_ORIGIN').encode() + b'\r\nWebSocket-Location: ' + location.encode() + b'\r\n\r\n'
    elif self.protocol_version == 76:
        handshake_reply = b'HTTP/1.1 101 WebSocket Protocol Handshake\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\nSec-WebSocket-Origin: ' + environ.get('HTTP_ORIGIN').encode() + b'\r\nSec-WebSocket-Protocol: ' + environ.get('HTTP_SEC_WEBSOCKET_PROTOCOL', 'default').encode() + b'\r\nSec-WebSocket-Location: ' + location.encode() + b'\r\n\r\n' + response
    else:
        raise ValueError('Unknown WebSocket protocol version.')
    sock.sendall(handshake_reply)
    return WebSocket(sock, environ, self.protocol_version)