from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_proper_allow_header_multiple_gets(self):

    class ThingsController(RestController):

        @expose()
        def get_all(self):
            return dict()

        @expose()
        def get(self):
            return dict()
    app = TestApp(make_app(ThingsController()))
    r = app.put('/123', status=405)
    assert r.status_int == 405
    assert r.headers['Allow'] == 'GET'