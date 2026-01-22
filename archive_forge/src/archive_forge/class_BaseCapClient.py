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
class BaseCapClient(object):

    def __init__(self, port):
        self.port = port

    def process_response(self, data):
        m = re.match('(.*)\t(.*)\n', to_unicode(data))
        if m is None:
            raise Exception('did not match')
        status, message = m.groups()
        if status == 'ok':
            return message
        else:
            raise CapError(message)