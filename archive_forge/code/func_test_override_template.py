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
def test_override_template(self):

    class RootController(object):

        @expose('foo.html')
        def index(self):
            override_template(None, content_type='text/plain')
            return 'Override'
    app = TestApp(Pecan(RootController()))
    r = app.get('/')
    assert r.status_int == 200
    assert b'Override' in r.body
    assert r.content_type == 'text/plain'