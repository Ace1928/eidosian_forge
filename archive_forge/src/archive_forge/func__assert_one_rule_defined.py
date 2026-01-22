import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _assert_one_rule_defined(self):
    r = self.get('/role_inferences/')
    rules = r.result['role_inferences']
    self.assertEqual(self.prior['id'], rules[0]['prior_role']['id'])
    self.assertEqual(self.implied1['id'], rules[0]['implies'][0]['id'])
    self.assertEqual(self.implied1['name'], rules[0]['implies'][0]['name'])
    self.assertEqual(1, len(rules[0]['implies']))