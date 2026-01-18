import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_secured_generic_controller(self):
    authorized = False

    class RootController(object):

        @classmethod
        def check_permissions(cls):
            return authorized

        @expose(generic=True)
        def index(self):
            return 'Index'

        @secure('check_permissions')
        @index.when(method='POST')
        def index_post(self):
            return 'I should not be allowed'

        @secure('check_permissions')
        @expose(generic=True)
        def secret(self):
            return 'I should not be allowed'
    app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))
    response = app.get('/')
    assert response.status_int == 200
    response = app.post('/', expect_errors=True)
    assert response.status_int == 401
    response = app.get('/secret/', expect_errors=True)
    assert response.status_int == 401