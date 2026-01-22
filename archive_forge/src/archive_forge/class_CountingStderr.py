from functools import reduce
import gc
import io
import locale  # system locale module, not tornado.locale
import logging
import operator
import textwrap
import sys
import unittest
import warnings
from tornado.httpclient import AsyncHTTPClient
from tornado.httpserver import HTTPServer
from tornado.netutil import Resolver
from tornado.options import define, add_parse_callback, options
class CountingStderr(io.IOBase):

    def __init__(self, real):
        self.real = real
        self.byte_count = 0

    def write(self, data):
        self.byte_count += len(data)
        return self.real.write(data)

    def flush(self):
        return self.real.flush()