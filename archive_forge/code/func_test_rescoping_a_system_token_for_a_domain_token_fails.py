import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_rescoping_a_system_token_for_a_domain_token_fails(self):
    ref = {'role': unit.new_role_ref()}
    system_role = self.post('/roles', body=ref).json_body['role']
    path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role['id']}
    self.put(path=path)
    auth_request_body = self.build_authentication_request(username=self.user['name'], password=self.user['password'], user_domain_id=self.domain['id'], system=True)
    response = self.v3_create_token(auth_request_body)
    self.assertValidSystemScopedTokenResponse(response)
    system_token = response.headers.get('X-Subject-Token')
    auth_request_body = self.build_authentication_request(token=system_token, domain_id=CONF.identity.default_domain_id)
    self.v3_create_token(auth_request_body, expected_status=http.client.FORBIDDEN)