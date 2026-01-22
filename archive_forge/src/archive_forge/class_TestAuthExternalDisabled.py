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
class TestAuthExternalDisabled(test_v3.RestfulTestCase):

    def config_overrides(self):
        super(TestAuthExternalDisabled, self).config_overrides()
        self.config_fixture.config(group='auth', methods=['password', 'token'])

    def test_remote_user_disabled(self):
        app = self.loadapp()
        remote_user = '%s@%s' % (self.user['name'], self.domain['name'])
        with app.test_client() as c:
            c.environ_base.update(self.build_external_auth_environ(remote_user))
            auth_data = self.build_authentication_request()
            c.post('/v3/auth/tokens', json=auth_data, expected_status_code=http.client.UNAUTHORIZED)