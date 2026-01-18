import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_restcontroller_with_hooks(self):

    class SomeHook(PecanHook):

        def before(self, state):
            state.response.headers['X-Testing'] = 'XYZ'

    class BaseController(rest.RestController):

        @expose()
        def delete(self, _id):
            return 'Deleting %s' % _id

    class RootController(BaseController, HookController):
        __hooks__ = [SomeHook()]

        @expose()
        def get_all(self):
            return 'Hello, World!'

        @staticmethod
        def static(cls):
            return 'static'

        @property
        def foo(self):
            return 'bar'

        def testing123(self):
            return 'bar'
        unhashable = [1, 'two', 3]
    app = TestApp(make_app(RootController()))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert response.headers['X-Testing'] == 'XYZ'
    response = app.delete('/100/')
    assert response.status_int == 200
    assert response.body == b'Deleting 100'
    assert response.headers['X-Testing'] == 'XYZ'