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
def test_thumbprint_verify_is_false_200(self):
    conf = copy.deepcopy(self._test_conf)
    conf['thumbprint_verify'] = False
    self.set_middleware(conf=conf)
    metadata = copy.deepcopy(self._default_metadata)
    metadata['access_project'].pop('id')
    roles = 'reader'
    metadata['roles'] = roles

    def mock_resp(request, context):
        return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, access_token=self._token, active=True, cert_thumb='this is an incorrectly thumbprint.', metadata=metadata)
    self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
    self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
    resp = self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
    self.assertEqual(FakeApp.SUCCESS, resp.body)
    self.assertEqual(resp.request.environ['HTTP_X_IDENTITY_STATUS'], 'Confirmed')
    self._check_env_value_domain_scope(resp.request.environ, self._user_id, self._user_name, self._user_domain_id, self._user_domain_name, self._project_domain_id, self._project_domain_name, roles, is_admin=False)