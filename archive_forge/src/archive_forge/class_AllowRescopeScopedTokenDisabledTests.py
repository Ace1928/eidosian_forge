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
class AllowRescopeScopedTokenDisabledTests(test_v3.RestfulTestCase):

    def config_overrides(self):
        super(AllowRescopeScopedTokenDisabledTests, self).config_overrides()
        self.config_fixture.config(group='token', allow_rescope_scoped_token=False)

    def test_rescoping_v3_to_v3_disabled(self):
        self.v3_create_token(self.build_authentication_request(token=self.get_scoped_token(), project_id=self.project_id), expected_status=http.client.FORBIDDEN)

    def test_rescoped_domain_token_disabled(self):
        self.domainA = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domainA['id'], self.domainA)
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domainA['id'])
        unscoped_token = self.get_requested_token(self.build_authentication_request(user_id=self.user['id'], password=self.user['password']))
        domain_scoped_token = self.get_requested_token(self.build_authentication_request(token=unscoped_token, domain_id=self.domainA['id']))
        self.v3_create_token(self.build_authentication_request(token=domain_scoped_token, project_id=self.project_id), expected_status=http.client.FORBIDDEN)