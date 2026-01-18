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
def test_project_scoped_token_catalog_attributes(self):
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    r = self.v3_create_token(auth_data)
    catalog = r.result['token']['catalog']
    self.assertEqual(1, len(catalog))
    catalog = catalog[0]
    self.assertEqual(self.service['id'], catalog['id'])
    self.assertEqual(self.service['name'], catalog['name'])
    self.assertEqual(self.service['type'], catalog['type'])
    endpoint = catalog['endpoints']
    self.assertEqual(1, len(endpoint))
    endpoint = endpoint[0]
    self.assertEqual(self.endpoint['id'], endpoint['id'])
    self.assertEqual(self.endpoint['interface'], endpoint['interface'])
    self.assertEqual(self.endpoint['region_id'], endpoint['region_id'])
    self.assertEqual(self.endpoint['url'], endpoint['url'])