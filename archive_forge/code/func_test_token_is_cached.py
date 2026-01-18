import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
def test_token_is_cached(self):
    context = auth_core.AuthContext(user_id=self.user['id'], methods=['password'])
    token = PROVIDERS.token_provider_api.issue_token(context['user_id'], context['methods'], project_id=self.project_id, auth_context=context)
    headers = {authorization.AUTH_TOKEN_HEADER: token.id.encode('utf-8')}
    with mock.patch.object(PROVIDERS.token_provider_api, 'validate_token', return_value=token) as token_mock:
        self._do_middleware_request(path='/v3/projects', method='get', headers=headers)
        token_mock.assert_called_once()