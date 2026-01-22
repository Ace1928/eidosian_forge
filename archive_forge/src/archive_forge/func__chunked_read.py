import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
def _chunked_read(self, rfile, length=None, use_readline=False):
    if self.should_send_hundred_continue:
        self.send_hundred_continue_response()
        self.is_hundred_continue_response_sent = True
    try:
        if length == 0:
            return b''
        if length and length < 0:
            length = None
        if use_readline:
            reader = self.rfile.readline
        else:
            reader = self.rfile.read
        response = []
        while self.chunk_length != 0:
            maxreadlen = self.chunk_length - self.position
            if length is not None and length < maxreadlen:
                maxreadlen = length
            if maxreadlen > 0:
                data = reader(maxreadlen)
                if not data:
                    self.chunk_length = 0
                    raise OSError('unexpected end of file while parsing chunked data')
                datalen = len(data)
                response.append(data)
                self.position += datalen
                if self.chunk_length == self.position:
                    rfile.readline()
                if length is not None:
                    length -= datalen
                    if length == 0:
                        break
                if use_readline and data[-1:] == b'\n':
                    break
            else:
                try:
                    self.chunk_length = int(rfile.readline().split(b';', 1)[0], 16)
                except ValueError as err:
                    raise ChunkReadError(err)
                self.position = 0
                if self.chunk_length == 0:
                    rfile.readline()
    except greenio.SSL.ZeroReturnError:
        pass
    return b''.join(response)