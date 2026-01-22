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
def _assert_initial_assignment_in_effective(self, response, user, project):
    entity = self.build_role_assignment_entity(project_id=project['id'], user_id=user['id'], role_id=self.role_list[0]['id'])
    self.assertRoleAssignmentInListResponse(response, entity)