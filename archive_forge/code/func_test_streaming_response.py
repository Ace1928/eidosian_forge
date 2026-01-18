import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
def test_streaming_response(self):

    class RootController(object):

        @expose(content_type='text/plain')
        def test(self, foo):
            if foo == 'stream':
                contents = BytesIO(b'stream')
                response.content_type = 'application/octet-stream'
                contents.seek(0, os.SEEK_END)
                response.content_length = contents.tell()
                contents.seek(0, os.SEEK_SET)
                response.app_iter = contents
                return response
            else:
                return 'plain text'
    app = TestApp(Pecan(RootController()))
    r = app.get('/test/stream')
    assert r.content_type == 'application/octet-stream'
    assert r.body == b'stream'
    r = app.get('/test/plain')
    assert r.content_type == 'text/plain'
    assert r.body == b'plain text'