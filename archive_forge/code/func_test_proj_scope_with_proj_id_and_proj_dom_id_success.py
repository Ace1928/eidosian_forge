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
def test_proj_scope_with_proj_id_and_proj_dom_id_success(self):
    env = {}
    env['SSL_CLIENT_I_DN'] = self.client_issuer
    env['HTTP_X_PROJECT_ID'] = self.project_id
    env['HTTP_X_PROJECT_DOMAIN_ID'] = self.domain_id
    env['SSL_CLIENT_USER_NAME'] = self.user['name']
    env['SSL_CLIENT_DOMAIN_NAME'] = self.domain_name
    self._load_mapping_rules(mapping_fixtures.MAPPING_WITH_USERNAME_AND_DOMAINNAME)
    req = self._do_middleware_request(extra_environ=env)
    context = req.environ.get(authorization.AUTH_CONTEXT_ENV)
    self._assert_tokenless_auth_context(context)
    request_context = req.environ.get(keystone_context.REQUEST_CONTEXT_ENV)
    self._assert_tokenless_request_context(request_context)