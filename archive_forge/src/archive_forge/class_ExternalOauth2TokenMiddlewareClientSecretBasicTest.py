import base64
import copy
import hashlib
import jwt.utils
import logging
import ssl
from testtools import matchers
import time
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import six
from six.moves import http_client
import testresources
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from keystonemiddleware.auth_token import _cache
from keystonemiddleware import external_oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
class ExternalOauth2TokenMiddlewareClientSecretBasicTest(BaseExternalOauth2TokenMiddlewareTest):

    def setUp(self):
        super(ExternalOauth2TokenMiddlewareClientSecretBasicTest, self).setUp()
        self._test_client_id = str(uuid.uuid4())
        self._test_client_secret = str(uuid.uuid4())
        self._auth_method = 'client_secret_basic'
        self._test_conf = get_config(introspect_endpoint=self._introspect_endpoint, audience=self._audience, auth_method=self._auth_method, client_id=self._test_client_id, client_secret=self._test_client_secret, thumbprint_verify=False, mapping_project_id='access_project.id', mapping_project_name='access_project.name', mapping_project_domain_id='access_project.domain.id', mapping_project_domain_name='access_project.domain.name', mapping_user_id='client_id', mapping_user_name='username', mapping_user_domain_id='user_domain.id', mapping_user_domain_name='user_domain.name', mapping_roles='roles')
        self._token = str(uuid.uuid4()) + '_user_token'
        self._user_id = str(uuid.uuid4()) + '_user_id'
        self._user_name = str(uuid.uuid4()) + '_user_name'
        self._user_domain_id = str(uuid.uuid4()) + '_user_domain_id'
        self._user_domain_name = str(uuid.uuid4()) + '_user_domain_name'
        self._project_id = str(uuid.uuid4()) + '_project_id'
        self._project_name = str(uuid.uuid4()) + '_project_name'
        self._project_domain_id = str(uuid.uuid4()) + 'project_domain_id'
        self._project_domain_name = str(uuid.uuid4()) + 'project_domain_name'
        self._roles = 'admin,member,reader'
        self._default_metadata = {'access_project': {'id': self._project_id, 'name': self._project_name, 'domain': {'id': self._project_domain_id, 'name': self._project_domain_name}}, 'user_domain': {'id': self._user_domain_id, 'name': self._user_domain_name}, 'roles': self._roles, 'client_id': self._user_id, 'username': self._user_name}
        self._clear_call_count = 0

    def test_basic_200(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        resp = self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        self._check_env_value_project_scope(resp.request.environ, self._user_id, self._user_name, self._user_domain_id, self._user_domain_name, self._project_id, self._project_name, self._project_domain_id, self._project_domain_name, self._roles)

    def test_domain_scope_200(self):
        conf = copy.deepcopy(self._test_conf)
        conf.pop('mapping_project_id')
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        resp = self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        self._check_env_value_domain_scope(resp.request.environ, self._user_id, self._user_name, self._user_domain_id, self._user_domain_name, self._project_domain_id, self._project_domain_name, self._roles)

    def test_system_scope_200(self):
        conf = copy.deepcopy(self._test_conf)
        conf.pop('mapping_project_id')
        conf['mapping_system_scope'] = 'system.all'
        self.set_middleware(conf=conf)
        self._default_metadata['system'] = {'all': True}

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata, system_scope=True)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        resp = self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        self._check_env_value_system_scope(resp.request.environ, self._user_id, self._user_name, self._user_domain_id, self._user_domain_name, self._roles)

    def test_process_response_401(self):
        conf = copy.deepcopy(self._test_conf)
        conf.pop('mapping_project_id')
        self.set_middleware(conf=conf)
        self.middleware = external_oauth2_token.ExternalAuth2Protocol(FakeOauth2TokenV3App(expected_env=self.expected_env, app_response_status_code=401), self.conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        resp = self.call_middleware(headers=get_authorization_header(self._token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertEqual(resp.headers.get('WWW-Authenticate'), 'Authorization OAuth 2.0 uri="%s"' % self._audience)