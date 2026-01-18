import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_single_blacklist_item(self):
    _stdout = StringIO()

    class RootController(object):

        @expose()
        def index(self):
            return 'Hello, World!'
    app = TestApp(make_app(RootController(), hooks=lambda: [RequestViewerHook(config={'blacklist': ['/']}, writer=_stdout)]))
    response = app.get('/')
    out = _stdout.getvalue()
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert out == ''