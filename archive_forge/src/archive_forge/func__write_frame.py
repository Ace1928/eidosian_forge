import abc
import asyncio
import base64
import hashlib
import os
import sys
import struct
import tornado
from urllib.parse import urlparse
import warnings
import zlib
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen, httpclient, httputil
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError, IOStream
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado import simple_httpclient
from tornado.queues import Queue
from tornado.tcpclient import TCPClient
from tornado.util import _websocket_mask
from typing import (
from types import TracebackType
def _write_frame(self, fin: bool, opcode: int, data: bytes, flags: int=0) -> 'Future[None]':
    data_len = len(data)
    if opcode & 8:
        if not fin:
            raise ValueError('control frames may not be fragmented')
        if data_len > 125:
            raise ValueError('control frame payloads may not exceed 125 bytes')
    if fin:
        finbit = self.FIN
    else:
        finbit = 0
    frame = struct.pack('B', finbit | opcode | flags)
    if self.mask_outgoing:
        mask_bit = 128
    else:
        mask_bit = 0
    if data_len < 126:
        frame += struct.pack('B', data_len | mask_bit)
    elif data_len <= 65535:
        frame += struct.pack('!BH', 126 | mask_bit, data_len)
    else:
        frame += struct.pack('!BQ', 127 | mask_bit, data_len)
    if self.mask_outgoing:
        mask = os.urandom(4)
        data = mask + _websocket_mask(mask, data)
    frame += data
    self._wire_bytes_out += len(frame)
    return self.stream.write(frame)