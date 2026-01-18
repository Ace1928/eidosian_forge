from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_role_assignments_directed_graph_of_implied_roles(self):
    """Test that a role can have multiple, different prior roles."""
    test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 6}, 'implied_roles': [{'role': 0, 'implied_roles': [1, 2]}, {'role': 1, 'implied_roles': [3, 4]}, {'role': 5, 'implied_roles': 4}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 5, 'project': 0}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 5, 'project': 0}, {'user': 0, 'role': 1, 'project': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 4, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 4, 'project': 0, 'indirect': {'role': 5}}]}]}
    test_data = self.execute_assignment_plan(test_plan)
    role_ids = PROVIDERS.assignment_api.get_roles_for_user_and_project(test_data['users'][0]['id'], test_data['projects'][0]['id'])
    self.assertThat(role_ids, matchers.HasLength(6))
    for x in range(0, 5):
        self.assertIn(test_data['roles'][x]['id'], role_ids)