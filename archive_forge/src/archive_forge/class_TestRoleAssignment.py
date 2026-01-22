from openstack.identity.v3 import role_assignment
from openstack.tests.unit import base
class TestRoleAssignment(base.TestCase):

    def test_basic(self):
        sot = role_assignment.RoleAssignment()
        self.assertEqual('role_assignment', sot.resource_key)
        self.assertEqual('role_assignments', sot.resources_key)
        self.assertEqual('/role_assignments', sot.base_path)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = role_assignment.RoleAssignment(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['scope'], sot.scope)
        self.assertEqual(EXAMPLE['user'], sot.user)
        self.assertEqual(EXAMPLE['group'], sot.group)