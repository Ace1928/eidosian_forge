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
def try_decode_UTF8(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return False
    except Exception as e:
        raise e