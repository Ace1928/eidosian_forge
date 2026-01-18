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
def test_user_and_group_roles_scoped_token(self):
    """Test correct roles are returned in scoped token.

        Test Plan:

        - Create a domain, with 1 project, 2 users (user1 and user2)
          and 2 groups (group1 and group2)
        - Make user1 a member of group1, user2 a member of group2
        - Create 8 roles, assigning them to each of the 8 combinations
          of users/groups on domain/project
        - Get a project scoped token for user1, checking that the right
          two roles are returned (one directly assigned, one by virtue
          of group membership)
        - Repeat this for a domain scoped token
        - Make user1 also a member of group2
        - Get another scoped token making sure the additional role
          shows up
        - User2 is just here as a spoiler, to make sure we don't get
          any roles uniquely assigned to it returned in any of our
          tokens

        """
    domainA = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domainA['id'], domainA)
    projectA = unit.new_project_ref(domain_id=domainA['id'])
    PROVIDERS.resource_api.create_project(projectA['id'], projectA)
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domainA['id'])
    user2 = unit.create_user(PROVIDERS.identity_api, domain_id=domainA['id'])
    group1 = unit.new_group_ref(domain_id=domainA['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    group2 = unit.new_group_ref(domain_id=domainA['id'])
    group2 = PROVIDERS.identity_api.create_group(group2)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
    PROVIDERS.identity_api.add_user_to_group(user2['id'], group2['id'])
    role_list = []
    for _ in range(8):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    PROVIDERS.assignment_api.create_grant(role_list[0]['id'], user_id=user1['id'], domain_id=domainA['id'])
    PROVIDERS.assignment_api.create_grant(role_list[1]['id'], user_id=user1['id'], project_id=projectA['id'])
    PROVIDERS.assignment_api.create_grant(role_list[2]['id'], user_id=user2['id'], domain_id=domainA['id'])
    PROVIDERS.assignment_api.create_grant(role_list[3]['id'], user_id=user2['id'], project_id=projectA['id'])
    PROVIDERS.assignment_api.create_grant(role_list[4]['id'], group_id=group1['id'], domain_id=domainA['id'])
    PROVIDERS.assignment_api.create_grant(role_list[5]['id'], group_id=group1['id'], project_id=projectA['id'])
    PROVIDERS.assignment_api.create_grant(role_list[6]['id'], group_id=group2['id'], domain_id=domainA['id'])
    PROVIDERS.assignment_api.create_grant(role_list[7]['id'], group_id=group2['id'], project_id=projectA['id'])
    auth_data = self.build_authentication_request(user_id=user1['id'], password=user1['password'], project_id=projectA['id'])
    r = self.v3_create_token(auth_data)
    token = self.assertValidScopedTokenResponse(r)
    roles_ids = []
    for ref in token['roles']:
        roles_ids.append(ref['id'])
    self.assertEqual(2, len(token['roles']))
    self.assertIn(role_list[1]['id'], roles_ids)
    self.assertIn(role_list[5]['id'], roles_ids)
    auth_data = self.build_authentication_request(user_id=user1['id'], password=user1['password'], domain_id=domainA['id'])
    r = self.v3_create_token(auth_data)
    token = self.assertValidScopedTokenResponse(r)
    roles_ids = []
    for ref in token['roles']:
        roles_ids.append(ref['id'])
    self.assertEqual(2, len(token['roles']))
    self.assertIn(role_list[0]['id'], roles_ids)
    self.assertIn(role_list[4]['id'], roles_ids)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group2['id'])
    auth_data = self.build_authentication_request(user_id=user1['id'], password=user1['password'], project_id=projectA['id'])
    r = self.v3_create_token(auth_data)
    token = self.assertValidScopedTokenResponse(r)
    roles_ids = []
    for ref in token['roles']:
        roles_ids.append(ref['id'])
    self.assertEqual(3, len(token['roles']))
    self.assertIn(role_list[1]['id'], roles_ids)
    self.assertIn(role_list[5]['id'], roles_ids)
    self.assertIn(role_list[7]['id'], roles_ids)