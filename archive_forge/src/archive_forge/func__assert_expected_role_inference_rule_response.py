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
def _assert_expected_role_inference_rule_response(self, expected_prior_id, expected_implied_id):
    url = '/roles/%s/implies/%s' % (expected_prior_id, expected_implied_id)
    response = self.get(url).json
    self.assertThat(response['links']['self'], matchers.EndsWith('/v3%s' % url))
    role_inference = response['role_inference']
    prior_role = role_inference['prior_role']
    self.assertEqual(expected_prior_id, prior_role['id'])
    self.assertIsNotNone(prior_role['name'])
    self.assertThat(prior_role['links']['self'], matchers.EndsWith('/v3/roles/%s' % expected_prior_id))
    implied_role = role_inference['implies']
    self.assertEqual(expected_implied_id, implied_role['id'])
    self.assertIsNotNone(implied_role['name'])
    self.assertThat(implied_role['links']['self'], matchers.EndsWith('/v3/roles/%s' % expected_implied_id))