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
def test_roles_subset(self):
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    PROVIDERS.assignment_api.create_grant(role_id=role['id'], user_id=self.user_id, project_id=self.project_id)
    ref = self.redelegated_trust_ref
    ref['expires_at'] = datetime.datetime.utcnow().replace(year=2032).strftime(unit.TIME_FORMAT)
    ref['roles'].append({'id': role['id']})
    r = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(r)
    role_id_set = set((r['id'] for r in ref['roles']))
    trust_role_id_set = set((r['id'] for r in trust['roles']))
    self.assertEqual(role_id_set, trust_role_id_set)
    trust_token = self._get_trust_token(trust)
    self.chained_trust_ref['expires_at'] = datetime.datetime.utcnow().replace(year=2028).strftime(unit.TIME_FORMAT)
    r = self.post('/OS-TRUST/trusts', body={'trust': self.chained_trust_ref}, token=trust_token)
    trust2 = self.assertValidTrustResponse(r)
    role_id_set1 = set((r['id'] for r in trust['roles']))
    role_id_set2 = set((r['id'] for r in trust2['roles']))
    self.assertThat(role_id_set1, matchers.GreaterThan(role_id_set2))