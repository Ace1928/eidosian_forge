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
def test_prioritized_hooks(self):
    run_hook = []

    class RootController(object):

        @expose()
        def index(self, req, resp):
            run_hook.append('inside')
            return 'Hello, World!'

    class SimpleHook(PecanHook):

        def __init__(self, id, priority=None):
            self.id = str(id)
            if priority:
                self.priority = priority

        def on_route(self, state):
            run_hook.append('on_route' + self.id)

        def before(self, state):
            run_hook.append('before' + self.id)

        def after(self, state):
            run_hook.append('after' + self.id)

        def on_error(self, state, e):
            run_hook.append('error' + self.id)
    papp = Pecan(RootController(), hooks=[SimpleHook(1, 3), SimpleHook(2, 2), SimpleHook(3, 1)], use_context_locals=False)
    app = TestApp(papp)
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert len(run_hook) == 10
    assert run_hook[0] == 'on_route3'
    assert run_hook[1] == 'on_route2'
    assert run_hook[2] == 'on_route1'
    assert run_hook[3] == 'before3'
    assert run_hook[4] == 'before2'
    assert run_hook[5] == 'before1'
    assert run_hook[6] == 'inside'
    assert run_hook[7] == 'after1'
    assert run_hook[8] == 'after2'
    assert run_hook[9] == 'after3'