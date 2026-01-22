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
def _assert_effective_role_for_implied_has_prior_in_links(self, response, user, project, prior_index, implied_index):
    prior_link = '/prior_roles/%(prior)s/implies/%(implied)s' % {'prior': self.role_list[prior_index]['id'], 'implied': self.role_list[implied_index]['id']}
    link = self.build_role_assignment_link(project_id=project['id'], user_id=user['id'], role_id=self.role_list[prior_index]['id'])
    entity = self.build_role_assignment_entity(link=link, project_id=project['id'], user_id=user['id'], role_id=self.role_list[implied_index]['id'], prior_link=prior_link)
    self.assertRoleAssignmentInListResponse(response, entity)