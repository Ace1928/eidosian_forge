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
class TestThreadLocalState(PecanTestCase):

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

    def test_request_state_cleanup(self):
        """
        After a request, the state local() should be totally clean
        except for state.app (so that objects don't leak between requests)
        """
        from pecan.core import state

        class RootController(object):

            @expose()
            def index(self):
                return '/'
        app = TestApp(Pecan(RootController()))
        r = app.get('/')
        assert r.status_int == 200
        assert r.body == b'/'
        assert state.__dict__ == {}