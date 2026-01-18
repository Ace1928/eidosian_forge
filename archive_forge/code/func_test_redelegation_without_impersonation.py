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
def test_redelegation_without_impersonation(self):
    self.redelegated_trust_ref['impersonation'] = False
    resp = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref}, expected_status=http.client.CREATED)
    trust = self.assertValidTrustResponse(resp)
    auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
    trust_token = self.get_requested_token(auth_data)
    trustee_user_2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
    trust_ref_2 = unit.new_trust_ref(trustor_user_id=self.trustee_user['id'], trustee_user_id=trustee_user_2['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id], allow_redelegation=False)
    resp = self.post('/OS-TRUST/trusts', body={'trust': trust_ref_2}, token=trust_token, expected_status=http.client.NOT_FOUND)