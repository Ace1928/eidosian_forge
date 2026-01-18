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
def test_on_error_response_hook(self):
    run_hook = []

    class RootController(object):

        @expose()
        def causeerror(self, req, resp):
            return [][1]

    class ErrorHook(PecanHook):

        def on_error(self, state, e):
            run_hook.append('error')
            r = webob.Response()
            r.text = 'on_error'
            return r
    app = TestApp(Pecan(RootController(), hooks=[ErrorHook()], use_context_locals=False))
    response = app.get('/causeerror')
    assert len(run_hook) == 1
    assert run_hook[0] == 'error'
    assert response.text == 'on_error'