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
def test_query_for_role_id_does_not_return_system_user_roles(self):
    system_role_id = self._create_new_role()
    member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
    self.put(member_url)
    path = '/role_assignments?role.id=%(role_id)s&user.id=%(user_id)s' % {'role_id': self.role_id, 'user_id': self.user['id']}
    response = self.get(path)
    self.assertValidRoleAssignmentListResponse(response, expected_length=1)