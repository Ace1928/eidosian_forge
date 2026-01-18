from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_simple_nested_rest(self):

    class BarController(RestController):

        @expose()
        def post(self):
            return 'BAR-POST'

        @expose()
        def delete(self, id_):
            return 'BAR-%s' % id_

    class FooController(RestController):
        bar = BarController()

        @expose()
        def post(self):
            return 'FOO-POST'

        @expose()
        def delete(self, id_):
            return 'FOO-%s' % id_

    class RootController(object):
        foo = FooController()
    app = TestApp(make_app(RootController()))
    r = app.post('/foo')
    assert r.status_int == 200
    assert r.body == b'FOO-POST'
    r = app.delete('/foo/1')
    assert r.status_int == 200
    assert r.body == b'FOO-1'
    r = app.post('/foo/bar')
    assert r.status_int == 200
    assert r.body == b'BAR-POST'
    r = app.delete('/foo/bar/2')
    assert r.status_int == 200
    assert r.body == b'BAR-2'