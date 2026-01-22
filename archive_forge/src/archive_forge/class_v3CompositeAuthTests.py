import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
class v3CompositeAuthTests(BaseAuthTokenMiddlewareTest, CommonCompositeAuthTests, testresources.ResourcedTestCase):
    """Test auth_token middleware with v3 token based composite auth.

    Execute the Composite auth class tests, but with the
    auth_token middleware configured to expect v3 tokens back from
    a keystone server.
    """
    resources = [('examples', client_fixtures.EXAMPLES_RESOURCE)]

    def setUp(self):
        super(v3CompositeAuthTests, self).setUp(auth_version='v3', fake_app=v3CompositeFakeApp)
        uuid_token_default = self.examples.v3_UUID_TOKEN_DEFAULT
        uuid_serv_token_default = self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT
        uuid_token_bind = self.examples.v3_UUID_TOKEN_BIND
        uuid_service_token_bind = self.examples.v3_UUID_SERVICE_TOKEN_BIND
        self.token_dict = {'uuid_token_default': uuid_token_default, 'uuid_service_token_default': uuid_serv_token_default, 'uuid_token_bind': uuid_token_bind, 'uuid_service_token_bind': uuid_service_token_bind}
        self.requests_mock.get(BASE_URI, json=VERSION_LIST_v3, status_code=300)
        self.requests_mock.post('%s/v2.0/tokens' % BASE_URI, text=FAKE_ADMIN_TOKEN)
        self.requests_mock.get('%s/v3/auth/tokens' % BASE_URI, text=self.token_response, headers={'X-Subject-Token': uuid.uuid4().hex})
        self.token_expected_env = dict(EXPECTED_V2_DEFAULT_ENV_RESPONSE)
        self.token_expected_env.update(EXPECTED_V3_DEFAULT_ENV_ADDITIONS)
        self.service_token_expected_env = dict(EXPECTED_V2_DEFAULT_SERVICE_ENV_RESPONSE)
        self.service_token_expected_env.update(EXPECTED_V3_DEFAULT_SERVICE_ENV_ADDITIONS)
        self.set_middleware()

    def token_response(self, request, context):
        auth_id = request.headers.get('X-Auth-Token')
        token_id = request.headers.get('X-Subject-Token')
        self.assertEqual(auth_id, FAKE_ADMIN_TOKEN_ID)
        status = 200
        response = ''
        if token_id == ERROR_TOKEN:
            msg = 'Network connection refused.'
            raise ksa_exceptions.ConnectFailure(msg)
        elif token_id == TIMEOUT_TOKEN:
            request_timeout_response(request, context)
        try:
            response = self.examples.JSON_TOKEN_RESPONSES[token_id]
        except KeyError:
            status = 404
        context.status_code = status
        return response