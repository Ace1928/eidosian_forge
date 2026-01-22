from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import (
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
class CapServer(TCPServer):

    @gen.coroutine
    def handle_stream(self, stream, address):
        data = (yield stream.read_until(b'\n'))
        data = to_unicode(data)
        if data == data.upper():
            stream.write(b'error\talready capitalized\n')
        else:
            stream.write(utf8('ok\t%s' % data.upper()))
        stream.close()