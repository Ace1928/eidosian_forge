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
def test_trusts_from_implied_role(self):
    self._create_three_roles()
    self._create_implied_role(self.role_list[0], self.role_list[1])
    self._create_implied_role(self.role_list[1], self.role_list[2])
    self._assign_top_role_to_user_on_project(self.user, self.project)
    trustee = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
    ref = unit.new_trust_ref(trustor_user_id=self.user['id'], trustee_user_id=trustee['id'], project_id=self.project['id'], role_ids=[self.role_list[0]['id']])
    r = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = r.result['trust']
    self.assertEqual(self.role_list[0]['id'], trust['roles'][0]['id'])
    self.assertThat(trust['roles'], matchers.HasLength(1))
    auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'], trust_id=trust['id'])
    r = self.v3_create_token(auth_data)
    token = r.result['token']
    self.assertThat(token['roles'], matchers.HasLength(len(self.role_list)))
    for role in token['roles']:
        self.assertIn(role, self.role_list)
    for role in self.role_list:
        self.assertIn(role, token['roles'])