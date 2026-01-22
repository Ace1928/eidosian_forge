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
class ExternalOauth2TokenMiddlewareClientSecretPostTest(BaseExternalOauth2TokenMiddlewareTest):

    def setUp(self):
        super(ExternalOauth2TokenMiddlewareClientSecretPostTest, self).setUp()
        self._test_client_id = str(uuid.uuid4())
        self._test_client_secret = str(uuid.uuid4())
        self._auth_method = 'client_secret_post'
        self._test_conf = get_config(introspect_endpoint=self._introspect_endpoint, audience=self._audience, auth_method=self._auth_method, client_id=self._test_client_id, client_secret=self._test_client_secret, thumbprint_verify=False, mapping_project_id='project_id', mapping_project_name='project_name', mapping_project_domain_id='domain_id', mapping_project_domain_name='domain_name', mapping_user_id='user', mapping_user_name='username', mapping_user_domain_id='user_domain.id', mapping_user_domain_name='user_domain.name', mapping_roles='roles')
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
        self._default_metadata = {'project_id': self._project_id, 'project_name': self._project_name, 'domain_id': self._project_domain_id, 'domain_name': self._project_domain_name, 'user_domain': {'id': self._user_domain_id, 'name': self._user_domain_name}, 'roles': self._roles, 'user': self._user_id, 'username': self._user_name}

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

    def test_process_request_no_access_token_in_header_401(self):
        conf = copy.deepcopy(self._test_conf)
        test_audience = 'https://test_audience'
        conf['audience'] = test_audience
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        resp = self.call_middleware(headers={}, expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertEqual(resp.headers.get('WWW-Authenticate'), 'Authorization OAuth 2.0 uri="%s"' % test_audience)

    def test_read_data_from_token_key_type_not_dict_403(self):
        conf = copy.deepcopy(self._test_conf)
        conf['mapping_user_id'] = 'user.id'
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=403, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    def test_read_data_from_token_key_not_fount_in_metadata_403(self):
        conf = copy.deepcopy(self._test_conf)
        conf['mapping_user_id'] = 'user_id'
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=403, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    def test_read_data_from_token_key_value_type_is_not_match_403(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)
        metadata = copy.deepcopy(self._default_metadata)
        metadata['user'] = {'id': str(uuid.uuid4()), 'name': 'testName'}

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=403, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    def test_read_data_from_token_key_config_error_is_not_dict_500(self):
        conf = copy.deepcopy(self._test_conf)
        conf['mapping_project_id'] = '..project_id'
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    def test_read_data_from_token_key_config_error_is_not_set_500(self):
        conf = copy.deepcopy(self._test_conf)
        conf.pop('mapping_roles')
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})