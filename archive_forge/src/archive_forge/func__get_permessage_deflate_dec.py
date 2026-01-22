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
def _get_permessage_deflate_dec(self, rsv1):
    options = self.extensions.get('permessage-deflate')
    if options is None or not rsv1:
        return None

    def _make():
        return zlib.decompressobj(-options.get('server_max_window_bits' if self.client else 'client_max_window_bits', zlib.MAX_WBITS))
    if options.get('server_no_context_takeover' if self.client else 'client_no_context_takeover'):
        return _make()
    else:
        if self._deflate_dec is None:
            self._deflate_dec = _make()
        return self._deflate_dec