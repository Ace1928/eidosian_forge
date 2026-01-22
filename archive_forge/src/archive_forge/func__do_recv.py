import asyncore
import binascii
import collections
import errno
import functools
import hashlib
import hmac
import math
import os
import pickle
import socket
import struct
import time
import futurist
from oslo_utils import excutils
from taskflow.engines.action_engine import executor as base
from taskflow import logging
from taskflow import task as ta
from taskflow.types import notifier as nt
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.utils import schema_utils as su
from taskflow.utils import threading_utils
def _do_recv(self, read_pipe=None):
    if read_pipe is None:
        read_pipe = self._read_pipe
    msg_capture = collections.deque(maxlen=1)
    msg_capture_func = lambda _from_who, msg_decoder_func: msg_capture.append(msg_decoder_func())
    reader = Reader(self.auth_key, msg_capture_func, msg_limit=1)
    try:
        maybe_msg_num = self._received + 1
        bytes_needed = reader.bytes_needed
        while True:
            blob = read_pipe.read(bytes_needed)
            if len(blob) != bytes_needed:
                raise EOFError('Read pipe closed while reading %s bytes for potential message %s' % (bytes_needed, maybe_msg_num))
            reader.feed(blob)
            bytes_needed = reader.bytes_needed
    except StopIteration:
        pass
    msg = msg_capture[0]
    self._received += 1
    return msg