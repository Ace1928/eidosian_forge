import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def valid_ident(s):
    m = IDENTIFIER.match(s)
    if not m:
        raise ValueError('Not a valid Python identifier: %r' % s)
    return True