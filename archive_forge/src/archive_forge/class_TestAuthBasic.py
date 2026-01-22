import base64
import os
import tempfile
from oslo_config import cfg
import webob
from oslo_middleware import basic_auth as auth
from oslotest import base as test_base
class TestAuthBasic(test_base.BaseTestCase):

    def setUp(self):
        super().setUp()

        @webob.dec.wsgify
        def fake_app(req):
            return webob.Response()
        self.fake_app = fake_app
        self.request = webob.Request.blank('/')

    def write_auth_file(self, data=None):
        if not data:
            data = '\n'
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(data)
            self.addCleanup(os.remove, f.name)
            return f.name

    def test_middleware_authenticate(self):
        auth_file = self.write_auth_file('myName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
        cfg.CONF.set_override('http_basic_auth_user_file', auth_file, group='oslo_middleware')
        self.middleware = auth.BasicAuthMiddleware(self.fake_app)
        self.request.environ['HTTP_AUTHORIZATION'] = 'Basic bXlOYW1lOm15UGFzc3dvcmQ='
        response = self.request.get_response(self.middleware)
        self.assertEqual('200 OK', response.status)

    def test_middleware_unauthenticated(self):
        auth_file = self.write_auth_file('myName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
        cfg.CONF.set_override('http_basic_auth_user_file', auth_file, group='oslo_middleware')
        self.middleware = auth.BasicAuthMiddleware(self.fake_app)
        response = self.request.get_response(self.middleware)
        self.assertEqual('401 Unauthorized', response.status)

    def test_authenticate(self):
        auth_file = self.write_auth_file('foo:bar\nmyName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
        self.assertEqual({'HTTP_X_USER': 'myName', 'HTTP_X_USER_NAME': 'myName'}, auth.authenticate(auth_file, 'myName', b'myPassword'))
        e = self.assertRaises(webob.exc.HTTPBadRequest, auth.authenticate, auth_file, 'foo', b'bar')
        self.assertEqual('Only bcrypt digested passwords are supported for foo', str(e))
        auth_file = auth_file + '.missing'
        e = self.assertRaises(webob.exc.HTTPBadRequest, auth.authenticate, auth_file, 'myName', b'myPassword')
        self.assertEqual('Problem reading auth file', str(e))

    def test_auth_entry(self):
        entry_pass = 'myName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm'
        entry_fail = 'foo:bar'
        self.assertEqual({'HTTP_X_USER': 'myName', 'HTTP_X_USER_NAME': 'myName'}, auth.auth_entry(entry_pass, b'myPassword'))
        ex = self.assertRaises(webob.exc.HTTPBadRequest, auth.auth_entry, entry_fail, b'bar')
        self.assertEqual('Only bcrypt digested passwords are supported for foo', str(ex))
        self.assertRaises(webob.exc.HTTPUnauthorized, auth.auth_entry, entry_pass, b'bar')

    def test_validate_auth_file(self):
        auth_file = self.write_auth_file('myName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
        auth.validate_auth_file(auth_file)
        auth_file = auth_file + '.missing'
        self.assertRaises(auth.ConfigInvalid, auth.validate_auth_file, auth_file)
        auth_file = self.write_auth_file('foo:bar\nmyName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
        self.assertRaises(webob.exc.HTTPBadRequest, auth.validate_auth_file, auth_file)

    def test_parse_token(self):
        token = base64.b64encode(b'myName:myPassword')
        self.assertEqual(('myName', b'myPassword'), auth.parse_token(token))
        token = str(token, encoding='utf-8')
        self.assertEqual(('myName', b'myPassword'), auth.parse_token(token))
        e = self.assertRaises(webob.exc.HTTPBadRequest, auth.parse_token, token[:-1])
        self.assertEqual('Could not decode authorization token', str(e))
        token = str(base64.b64encode(b'myNamemyPassword'), encoding='utf-8')
        e = self.assertRaises(webob.exc.HTTPBadRequest, auth.parse_token, token[:-1])
        self.assertEqual('Could not decode authorization token', str(e))

    def test_parse_header(self):
        auth_value = 'Basic bXlOYW1lOm15UGFzc3dvcmQ='
        self.assertEqual('bXlOYW1lOm15UGFzc3dvcmQ=', auth.parse_header({'HTTP_AUTHORIZATION': auth_value}))
        e = self.assertRaises(webob.exc.HTTPUnauthorized, auth.parse_header, {})
        e = self.assertRaises(webob.exc.HTTPBadRequest, auth.parse_header, {'HTTP_AUTHORIZATION': 'Basic'})
        self.assertEqual('Could not parse Authorization header', str(e))
        digest_value = 'Digest username="myName" nonce="foobar"'
        e = self.assertRaises(webob.exc.HTTPBadRequest, auth.parse_header, {'HTTP_AUTHORIZATION': digest_value})
        self.assertEqual('Unsupported authorization type "Digest"', str(e))