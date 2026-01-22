import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
class TestSocketWrapper:

    def __init__(self, sock):
        self.__sock = sock

    def recv_all(self, n):
        buf = b''
        while len(buf) < n:
            data = self.recv(n - len(buf))
            if data == b'':
                raise ConnectionAbortedError
            buf += data
        return buf

    def starttls(self, ssl_context, *, server_side=False, server_hostname=None, do_handshake_on_connect=True):
        assert isinstance(ssl_context, ssl.SSLContext)
        ssl_sock = ssl_context.wrap_socket(self.__sock, server_side=server_side, server_hostname=server_hostname, do_handshake_on_connect=do_handshake_on_connect)
        if server_side:
            ssl_sock.do_handshake()
        self.__sock.close()
        self.__sock = ssl_sock

    def __getattr__(self, name):
        return getattr(self.__sock, name)

    def __repr__(self):
        return '<{} {!r}>'.format(type(self).__name__, self.__sock)