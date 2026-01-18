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
def test_with_multiple_users_and_invalid_credentials(self):
    """Prevent logging in with someone else's credentials.

        It's very easy to forget to limit the credentials query by user.
        Let's just test it for a sanity check.
        """
    self._make_credentials('totp', count=3)
    new_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
    PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=new_user['id'], project_id=self.project['id'])
    user2_creds = self._make_credentials('totp', count=1, user_id=new_user['id'])
    user_id = self.default_domain_user['id']
    secret = user2_creds[-1]['blob']
    auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0], user_id=user_id)
    self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)