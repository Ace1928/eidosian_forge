import os
import sys
import time
import random
import os.path
import platform
import warnings
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import libcloud.security
from libcloud.http import LibcloudConnection
from libcloud.test import unittest, no_network
from libcloud.utils.py3 import reload, httplib, assertRaisesRegex
class MockHTTPServerRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path in ['/test']:
            self.send_response(requests.codes.ok)
            self.end_headers()
        if self.path in ['/test-timeout']:
            time.sleep(1)
            self.send_response(requests.codes.ok)
            self.end_headers()
        elif self.path in ['/test/prepared-request-1', '/test/prepared-request-2']:
            headers = dict(self.headers)
            assert 'Content-Length' not in headers
            self.connection.setblocking(0)
            body = self.rfile.read(1)
            assert body is None
            self.send_response(requests.codes.ok)
            self.end_headers()
            self.wfile.write(self.path.encode('utf-8'))
        elif self.path == '/test/prepared-request-3':
            headers = dict(self.headers)
            assert int(headers['Content-Length']) == 9
            body = self.rfile.read(int(headers['Content-Length']))
            assert body == b'test body'
            self.send_response(requests.codes.ok)
            self.end_headers()
            self.wfile.write(self.path.encode('utf-8'))
        else:
            self.send_response(requests.codes.internal_server_error)
            self.end_headers()