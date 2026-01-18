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
def test_remove_all_roles_from_scope_result_in_404(self):
    new_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    path = '/projects/%s/users/%s/roles/%s' % (self.project['id'], new_user['id'], self.role['id'])
    self.put(path=path)
    auth_data = self.build_authentication_request(user_id=new_user['id'], password=new_user['password'], project_id=self.project['id'])
    subject_token_id = self.v3_create_token(auth_data).headers.get('X-Subject-Token')
    headers = {'X-Subject-Token': subject_token_id}
    r = self.get('/auth/tokens', headers=headers)
    self.assertValidProjectScopedTokenResponse(r)
    path = '/projects/%s/users/%s/roles/%s' % (self.project['id'], new_user['id'], self.role['id'])
    self.delete(path=path)
    self.get('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)