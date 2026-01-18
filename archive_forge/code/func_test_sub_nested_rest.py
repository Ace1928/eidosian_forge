from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_sub_nested_rest(self):

    class BazsController(RestController):
        data = [[['zero-zero-zero']]]

        @expose()
        def get_one(self, foo_id, bar_id, id):
            return self.data[int(foo_id)][int(bar_id)][int(id)]

    class BarsController(RestController):
        data = [['zero-zero']]
        bazs = BazsController()

        @expose()
        def get_one(self, foo_id, id):
            return self.data[int(foo_id)][int(id)]

    class FoosController(RestController):
        data = ['zero']
        bars = BarsController()

        @expose()
        def get_one(self, id):
            return self.data[int(id)]

    class RootController(object):
        foos = FoosController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foos/0/bars/0/bazs/0')
    assert r.status_int == 200
    assert r.body == b'zero-zero-zero'