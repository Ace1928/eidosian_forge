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
def test_not_socket(self):
    token = self.examples.v3_OAUTH2_CREDENTIAL
    resp = super(Oauth2mTlsTokenMiddlewareTest, self).call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(None)})
    self.assertUnauthorizedResp(resp)
    self.assertIn('Unable to obtain the client certificate.', self.logger.output)