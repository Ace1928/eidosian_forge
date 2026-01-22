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
class TestFetchRevocationList(test_v3.RestfulTestCase):
    """Test fetch token revocation list on the v3 Identity API."""

    def config_overrides(self):
        super(TestFetchRevocationList, self).config_overrides()
        self.config_fixture.config(group='token', revoke_by_id=True)

    def test_get_ids_no_tokens_returns_forbidden(self):
        self.get('/auth/tokens/OS-PKI/revoked', expected_status=http.client.FORBIDDEN)

    def test_head_ids_no_tokens_returns_forbidden(self):
        self.head('/auth/tokens/OS-PKI/revoked', expected_status=http.client.FORBIDDEN)