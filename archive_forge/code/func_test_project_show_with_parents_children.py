from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_project_show_with_parents_children(self):
    output = self.openstack('project show --parents --children --domain %(domain)s %(name)s' % {'domain': self.domain_name, 'name': self.project_name}, parse_output=True)
    for attr_name in self.PROJECT_FIELDS + ['parents', 'subtree']:
        self.assertIn(attr_name, output)
    self.assertEqual(self.project_name, output.get('name'))