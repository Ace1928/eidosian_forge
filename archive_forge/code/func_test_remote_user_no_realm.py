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
def test_remote_user_no_realm(self):
    app = self.loadapp()
    auth_contexts = []

    def new_init(self, *args, **kwargs):
        super(auth.core.AuthContext, self).__init__(*args, **kwargs)
        auth_contexts.append(self)
    self.useFixture(fixtures.MockPatch('keystone.auth.core.AuthContext.__init__', new_init))
    with app.test_client() as c:
        c.environ_base.update(self.build_external_auth_environ(self.default_domain_user['name']))
        auth_req = self.build_authentication_request()
        c.post('/v3/auth/tokens', json=auth_req)
        self.assertEqual(self.default_domain_user['id'], auth_contexts[-1]['user_id'])
    user = {'name': 'myname@mydivision'}
    PROVIDERS.identity_api.update_user(self.default_domain_user['id'], user)
    with app.test_client() as c:
        c.environ_base.update(self.build_external_auth_environ(user['name']))
        auth_req = self.build_authentication_request()
        c.post('/v3/auth/tokens', json=auth_req)
        self.assertEqual(self.default_domain_user['id'], auth_contexts[-1]['user_id'])
    self.assertEqual(self.default_domain_user['id'], auth_contexts[-1]['user_id'])