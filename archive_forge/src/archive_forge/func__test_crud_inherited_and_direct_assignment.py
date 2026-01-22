from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _test_crud_inherited_and_direct_assignment(self, **kwargs):
    """Test inherited and direct assignments for the actor and target.

        Ensure it is possible to create both inherited and direct role
        assignments for the same actor on the same target. The actor and the
        target are specified in the kwargs as ('user_id' or 'group_id') and
        ('project_id' or 'domain_id'), respectively.

        """
    role = unit.new_role_ref()
    role = PROVIDERS.role_api.create_role(role['id'], role)
    assignment_entity = {'role_id': role['id']}
    assignment_entity.update(kwargs)
    direct_assignment_entity = assignment_entity.copy()
    inherited_assignment_entity = assignment_entity.copy()
    inherited_assignment_entity['inherited_to_projects'] = 'projects'
    PROVIDERS.assignment_api.create_grant(inherited_to_projects=False, **assignment_entity)
    grants = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
    self.assertThat(grants, matchers.HasLength(1))
    self.assertIn(direct_assignment_entity, grants)
    PROVIDERS.assignment_api.create_grant(inherited_to_projects=True, **assignment_entity)
    grants = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
    self.assertThat(grants, matchers.HasLength(2))
    self.assertIn(direct_assignment_entity, grants)
    self.assertIn(inherited_assignment_entity, grants)
    PROVIDERS.assignment_api.delete_grant(inherited_to_projects=False, **assignment_entity)
    PROVIDERS.assignment_api.delete_grant(inherited_to_projects=True, **assignment_entity)
    grants = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
    self.assertEqual([], grants)