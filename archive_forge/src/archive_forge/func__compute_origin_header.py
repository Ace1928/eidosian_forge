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
def _compute_origin_header(self, url):
    origin = urlparse.urlparse(url)
    if origin.scheme == 'wss':
        return 'https://%s:%s' % (origin.hostname, origin.port)
    else:
        return 'http://%s:%s' % (origin.hostname, origin.port)