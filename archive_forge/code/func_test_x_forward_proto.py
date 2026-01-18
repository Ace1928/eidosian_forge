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
def test_x_forward_proto(self):

    class ChildController(object):

        @expose()
        def index(self):
            redirect('/testing')

    class RootController(object):

        @expose()
        def index(self):
            redirect('/testing')

        @expose()
        def testing(self):
            return 'it worked!'
        child = ChildController()
    app = TestApp(make_app(RootController(), debug=True))
    res = app.get('/child', extra_environ=dict(HTTP_X_FORWARDED_PROTO='https'))
    assert res.status_int == 302
    assert res.location == 'https://localhost/child/'
    assert res.request.environ['HTTP_X_FORWARDED_PROTO'] == 'https'