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
def test_root_role_as_implied_role_forbidden(self):
    """Test root role is forbidden to be set as an implied role.

        Create 2 roles that are prohibited from being an implied role.
        Create 1 additional role which should be accepted as an implied
        role. Assure the prohibited role names cannot be set as an implied
        role. Assure the accepted role name which is not a member of the
        prohibited implied role list can be successfully set an implied
        role.
        """
    prohibited_name1 = 'root1'
    prohibited_name2 = 'root2'
    accepted_name1 = 'implied1'
    prohibited_names = [prohibited_name1, prohibited_name2]
    self.config_fixture.config(group='assignment', prohibited_implied_role=prohibited_names)
    prior_role = self._create_role()
    prohibited_role1 = self._create_named_role(prohibited_name1)
    url = '/roles/{prior_role_id}/implies/{implied_role_id}'.format(prior_role_id=prior_role['id'], implied_role_id=prohibited_role1['id'])
    self.put(url, expected_status=http.client.FORBIDDEN)
    prohibited_role2 = self._create_named_role(prohibited_name2)
    url = '/roles/{prior_role_id}/implies/{implied_role_id}'.format(prior_role_id=prior_role['id'], implied_role_id=prohibited_role2['id'])
    self.put(url, expected_status=http.client.FORBIDDEN)
    accepted_role1 = self._create_named_role(accepted_name1)
    url = '/roles/{prior_role_id}/implies/{implied_role_id}'.format(prior_role_id=prior_role['id'], implied_role_id=accepted_role1['id'])
    self.put(url, expected_status=http.client.CREATED)