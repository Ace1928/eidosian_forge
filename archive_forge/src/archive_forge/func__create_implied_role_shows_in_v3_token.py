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
def _create_implied_role_shows_in_v3_token(self, is_domain):
    token_roles = self._get_scoped_token_roles(is_domain)
    self.assertEqual(1, len(token_roles))
    prior = token_roles[0]['id']
    implied1 = self._create_implied_role(prior)
    token_roles = self._get_scoped_token_roles(is_domain)
    self.assertEqual(2, len(token_roles))
    implied2 = self._create_implied_role(prior)
    token_roles = self._get_scoped_token_roles(is_domain)
    self.assertEqual(3, len(token_roles))
    token_role_ids = [role['id'] for role in token_roles]
    self.assertIn(prior, token_role_ids)
    self.assertIn(implied1['id'], token_role_ids)
    self.assertIn(implied2['id'], token_role_ids)