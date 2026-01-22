import functools
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
import http.client
import OpenSSL.SSL
import pytest
import requests
import trustme
from .._compat import bton, ntob, ntou
from .._compat import IS_ABOVE_OPENSSL10, IS_CI, IS_PYPY
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS
from ..server import HTTPServer, get_ssl_adapter_class
from ..testing import (
from ..wsgi import Gateway_10
class HelloWorldGateway(Gateway_10):
    """Gateway responding with Hello World to root URI."""

    def respond(self):
        """Respond with dummy content via HTTP."""
        req = self.req
        req_uri = bton(req.uri)
        if req_uri == '/':
            req.status = b'200 OK'
            req.ensure_headers_sent()
            req.write(b'Hello world!')
            return
        if req_uri == '/env':
            req.status = b'200 OK'
            req.ensure_headers_sent()
            env = self.get_environ()
            env.pop('wsgi.errors')
            env.pop('wsgi.input')
            print(env)
            req.write(json.dumps(env).encode('utf-8'))
            return
        return super(HelloWorldGateway, self).respond()