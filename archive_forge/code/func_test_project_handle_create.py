from unittest import mock
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.keystone import project
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_project_handle_create(self):
    mock_project = self._get_mock_project()
    self.projects.create.return_value = mock_project
    self.assertEqual('test_project_1', self.test_project.properties.get(project.KeystoneProject.NAME))
    self.assertEqual('Test project', self.test_project.properties.get(project.KeystoneProject.DESCRIPTION))
    self.assertEqual('default', self.test_project.properties.get(project.KeystoneProject.DOMAIN))
    self.assertEqual(True, self.test_project.properties.get(project.KeystoneProject.ENABLED))
    self.assertEqual('my_father', self.test_project.properties.get(project.KeystoneProject.PARENT))
    self.assertEqual(['label', 'insignia'], self.test_project.properties.get(project.KeystoneProject.TAGS))
    self.test_project.handle_create()
    self.projects.create.assert_called_once_with(name='test_project_1', description='Test project', domain='default', enabled=True, parent='my_father', tags=['label', 'insignia'])
    self.assertEqual(mock_project.id, self.test_project.resource_id)