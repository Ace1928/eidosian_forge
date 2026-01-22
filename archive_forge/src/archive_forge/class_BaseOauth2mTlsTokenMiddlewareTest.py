import http.client as http_client
import json
import logging
import ssl
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import testresources
from keystoneauth1 import access
from keystoneauth1 import exceptions as ksa_exceptions
from keystonemiddleware import oauth2_mtls_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit.test_oauth2_token_middleware \
from keystonemiddleware.tests.unit.test_oauth2_token_middleware \
from keystonemiddleware.tests.unit import utils
class BaseOauth2mTlsTokenMiddlewareTest(base.BaseAuthTokenTestCase):

    def setUp(self, expected_env=None, auth_version=None, fake_app=None):
        cfg.CONF.clear()
        super(BaseOauth2mTlsTokenMiddlewareTest, self).setUp()
        self.logger = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
        self.useFixture(fixtures.MockPatchObject(oauth2_mtls_token.OAuth2mTlsProtocol, '_create_oslo_cache', return_value=FakeOsloCache))
        self.expected_env = expected_env or dict()
        self.fake_app = fake_app or FakeApp
        self.middleware = None
        self.conf = {'identity_uri': 'https://keystone.example.com:1234/testadmin/', 'auth_version': auth_version, 'www_authenticate_uri': 'https://keystone.example.com:1234', 'admin_user': uuid.uuid4().hex}
        self.auth_version = auth_version

    def call_middleware(self, **kwargs):
        return self.call(self.middleware, **kwargs)

    def set_middleware(self, expected_env=None, conf=None):
        """Configure the class ready to call the oauth2_token middleware.

        Set up the various fake items needed to run the middleware.
        Individual tests that need to further refine these can call this
        function to override the class defaults.

        """
        if conf:
            self.conf.update(conf)
        if expected_env:
            self.expected_env.update(expected_env)
        self.middleware = oauth2_mtls_token.OAuth2mTlsProtocol(self.fake_app(self.expected_env), self.conf)

    def call(self, middleware, method='GET', path='/', headers=None, expected_status=http_client.OK, expected_body_string=None, **kwargs):
        req = webob.Request.blank(path, **kwargs)
        req.method = method
        for k, v in (headers or {}).items():
            req.headers[k] = v
        resp = req.get_response(middleware)
        self.assertEqual(expected_status, resp.status_int)
        if expected_body_string:
            self.assertIn(expected_body_string, str(resp.body))
        resp.request = req
        return resp

    def assertUnauthorizedResp(self, resp):
        error = json.loads(resp.body)
        self.assertEqual('Keystone uri="https://keystone.example.com:1234"', resp.headers['WWW-Authenticate'])
        self.assertEqual('Keystone uri="%s"' % self.conf.get('www_authenticate_uri'), resp.headers['WWW-Authenticate'])
        self.assertEqual('Unauthorized', error.get('error').get('title'))
        self.assertEqual('The request you have made requires authentication.', error.get('error').get('message'))