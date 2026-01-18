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
def test_thread_local_dir(self):
    """
        Threadlocal proxies for request and response should properly
        proxy ``dir()`` calls to the underlying webob class.
        """

    class RootController(object):

        @expose()
        def index(self):
            assert 'method' in dir(request)
            assert 'status' in dir(response)
            return '/'
    app = TestApp(Pecan(RootController()))
    r = app.get('/')
    assert r.status_int == 200
    assert r.body == b'/'