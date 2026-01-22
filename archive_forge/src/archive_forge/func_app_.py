import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
@property
def app_(self):

    class OthersController(object):

        @expose()
        def index(self, req, resp):
            return 'OTHERS'

        @expose()
        def echo(self, req, resp, value):
            return str(value)

    class ThingsController(RestController):
        data = ['zero', 'one', 'two', 'three']
        _custom_actions = {'count': ['GET'], 'length': ['GET', 'POST']}
        others = OthersController()

        @expose()
        def get_one(self, req, resp, id):
            return self.data[int(id)]

        @expose('json')
        def get_all(self, req, resp):
            return dict(items=self.data)

        @expose()
        def length(self, req, resp, id, value=None):
            length = len(self.data[int(id)])
            if value:
                length += len(value)
            return str(length)

        @expose()
        def post(self, req, resp, value):
            self.data.append(value)
            resp.status = 302
            return 'CREATED'

        @expose()
        def edit(self, req, resp, id):
            return 'EDIT %s' % self.data[int(id)]

        @expose()
        def put(self, req, resp, id, value):
            self.data[int(id)] = value
            return 'UPDATED'

        @expose()
        def get_delete(self, req, resp, id):
            return 'DELETE %s' % self.data[int(id)]

        @expose()
        def delete(self, req, resp, id):
            del self.data[int(id)]
            return 'DELETED'

        @expose()
        def trace(self, req, resp):
            return 'TRACE'

        @expose()
        def post_options(self, req, resp):
            return 'OPTIONS'

        @expose()
        def options(self, req, resp):
            abort(500)

        @expose()
        def other(self, req, resp):
            abort(500)

    class RootController(object):
        things = ThingsController()
    return TestApp(Pecan(RootController(), use_context_locals=False))