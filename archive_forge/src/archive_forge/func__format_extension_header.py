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
def _format_extension_header(self, parsed_extensions):
    if not parsed_extensions:
        return None
    parts = []
    for name, config in parsed_extensions.items():
        ext_parts = [name.encode()]
        for key, value in config.items():
            if value is False:
                pass
            elif value is True:
                ext_parts.append(key.encode())
            else:
                ext_parts.append(('%s=%s' % (key, str(value))).encode())
        parts.append(b'; '.join(ext_parts))
    return b', '.join(parts)