import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
class TestSecure(PecanTestCase):

    def test_simple_secure(self):
        authorized = False

        class SecretController(SecureController):

            @expose()
            def index(self):
                return 'Index'

            @expose()
            @unlocked
            def allowed(self):
                return 'Allowed!'

            @classmethod
            def check_permissions(cls):
                return authorized

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'

            @expose()
            @secure(lambda: False)
            def locked(self):
                return 'No dice!'

            @expose()
            @secure(lambda: True)
            def unlocked(self):
                return 'Sure thing'
            secret = SecretController()
        app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))
        response = app.get('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        response = app.get('/unlocked')
        assert response.status_int == 200
        assert response.body == b'Sure thing'
        response = app.get('/locked', expect_errors=True)
        assert response.status_int == 401
        response = app.get('/secret/', expect_errors=True)
        assert response.status_int == 401
        response = app.get('/secret/allowed')
        assert response.status_int == 200
        assert response.body == b'Allowed!'

    def test_unlocked_attribute(self):

        class AuthorizedSubController(object):

            @expose()
            def index(self):
                return 'Index'

            @expose()
            def allowed(self):
                return 'Allowed!'

        class SecretController(SecureController):

            @expose()
            def index(self):
                return 'Index'

            @expose()
            @unlocked
            def allowed(self):
                return 'Allowed!'
            authorized = unlocked(AuthorizedSubController())

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'

            @expose()
            @secure(lambda: False)
            def locked(self):
                return 'No dice!'

            @expose()
            @secure(lambda: True)
            def unlocked(self):
                return 'Sure thing'
            secret = SecretController()
        app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))
        response = app.get('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        response = app.get('/unlocked')
        assert response.status_int == 200
        assert response.body == b'Sure thing'
        response = app.get('/locked', expect_errors=True)
        assert response.status_int == 401
        response = app.get('/secret/', expect_errors=True)
        assert response.status_int == 401
        response = app.get('/secret/allowed')
        assert response.status_int == 200
        assert response.body == b'Allowed!'
        response = app.get('/secret/authorized/')
        assert response.status_int == 200
        assert response.body == b'Index'
        response = app.get('/secret/authorized/allowed')
        assert response.status_int == 200
        assert response.body == b'Allowed!'

    def test_secure_attribute(self):
        authorized = False

        class SubController(object):

            @expose()
            def index(self):
                return 'Hello from sub!'

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello from root!'
            sub = secure(SubController(), lambda: authorized)
        app = TestApp(make_app(RootController()))
        response = app.get('/')
        assert response.status_int == 200
        assert response.body == b'Hello from root!'
        response = app.get('/sub/', expect_errors=True)
        assert response.status_int == 401
        authorized = True
        response = app.get('/sub/')
        assert response.status_int == 200
        assert response.body == b'Hello from sub!'

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

    def test_secured_generic_controller_lambda(self):
        authorized = False

        class RootController(object):

            @expose(generic=True)
            def index(self):
                return 'Index'

            @secure(lambda: authorized)
            @index.when(method='POST')
            def index_post(self):
                return 'I should not be allowed'

            @secure(lambda: authorized)
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

    def test_secured_generic_controller_secure_attribute(self):
        authorized = False

        class SecureController(object):

            @expose(generic=True)
            def index(self):
                return 'I should not be allowed'

            @index.when(method='POST')
            def index_post(self):
                return 'I should not be allowed'

            @expose(generic=True)
            def secret(self):
                return 'I should not be allowed'

        class RootController(object):
            sub = secure(SecureController(), lambda: authorized)
        app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))
        response = app.get('/sub/', expect_errors=True)
        assert response.status_int == 401
        response = app.post('/sub/', expect_errors=True)
        assert response.status_int == 401
        response = app.get('/sub/secret/', expect_errors=True)
        assert response.status_int == 401

    def test_secured_generic_controller_secure_attribute_with_unlocked(self):

        class RootController(SecureController):

            @unlocked
            @expose(generic=True)
            def index(self):
                return 'Unlocked!'

            @unlocked
            @index.when(method='POST')
            def index_post(self):
                return 'Unlocked!'

            @expose(generic=True)
            def secret(self):
                return 'I should not be allowed'
        app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))
        response = app.get('/')
        assert response.status_int == 200
        response = app.post('/')
        assert response.status_int == 200
        response = app.get('/secret/', expect_errors=True)
        assert response.status_int == 401

    def test_state_attribute(self):
        from pecan.secure import Any, Protected
        assert repr(Any) == '<SecureState Any>'
        assert bool(Any) is False
        assert repr(Protected) == '<SecureState Protected>'
        assert bool(Protected) is True

    def test_secure_obj_only_failure(self):

        class Foo(object):
            pass
        try:
            secure(Foo())
        except Exception as e:
            assert isinstance(e, TypeError)