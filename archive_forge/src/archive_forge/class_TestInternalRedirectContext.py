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
class TestInternalRedirectContext(PecanTestCase):

    @property
    def app_(self):

        class RootController(object):

            @expose()
            def redirect_with_context(self):
                request.context['foo'] = 'bar'
                redirect('/testing')

            @expose()
            def internal_with_context(self):
                request.context['foo'] = 'bar'
                redirect('/testing', internal=True)

            @expose('json')
            def testing(self):
                return request.context
        return TestApp(make_app(RootController(), debug=False))

    def test_internal_with_request_context(self):
        r = self.app_.get('/internal_with_context')
        assert r.status_int == 200
        assert json.loads(r.body.decode()) == {'foo': 'bar'}

    def test_context_does_not_bleed(self):
        r = self.app_.get('/redirect_with_context').follow()
        assert r.status_int == 200
        assert json.loads(r.body.decode()) == {}