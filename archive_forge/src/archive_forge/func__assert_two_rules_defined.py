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
def _assert_two_rules_defined(self):
    r = self.get('/role_inferences/')
    rules = r.result['role_inferences']
    self.assertEqual(self.prior['id'], rules[0]['prior_role']['id'])
    self.assertEqual(2, len(rules[0]['implies']))
    implied_ids = [implied['id'] for implied in rules[0]['implies']]
    implied_names = [implied['name'] for implied in rules[0]['implies']]
    self.assertIn(self.implied1['id'], implied_ids)
    self.assertIn(self.implied2['id'], implied_ids)
    self.assertIn(self.implied1['name'], implied_names)
    self.assertIn(self.implied2['name'], implied_names)