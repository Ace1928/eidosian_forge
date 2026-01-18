from unittest import mock
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.keystone import project
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_project_handle_create_default(self):
    values = {project.KeystoneProject.NAME: None, project.KeystoneProject.DESCRIPTION: self._get_property_schema_value_default(project.KeystoneProject.DESCRIPTION), project.KeystoneProject.DOMAIN: self._get_property_schema_value_default(project.KeystoneProject.DOMAIN), project.KeystoneProject.ENABLED: self._get_property_schema_value_default(project.KeystoneProject.ENABLED), project.KeystoneProject.PARENT: self._get_property_schema_value_default(project.KeystoneProject.PARENT), project.KeystoneProject.TAGS: self._get_property_schema_value_default(project.KeystoneProject.TAGS)}

    def _side_effect(key):
        return values[key]
    mock_project = self._get_mock_project()
    self.projects.create.return_value = mock_project
    self.test_project.properties = mock.MagicMock()
    self.test_project.properties.get.side_effect = _side_effect
    self.test_project.properties.__getitem__.side_effect = _side_effect
    self.test_project.physical_resource_name = mock.MagicMock()
    self.test_project.physical_resource_name.return_value = 'foo'
    self.assertEqual(None, self.test_project.properties.get(project.KeystoneProject.NAME))
    self.assertEqual('', self.test_project.properties.get(project.KeystoneProject.DESCRIPTION))
    self.assertEqual('default', self.test_project.properties.get(project.KeystoneProject.DOMAIN))
    self.assertEqual(True, self.test_project.properties.get(project.KeystoneProject.ENABLED))
    self.assertIsNone(self.test_project.properties.get(project.KeystoneProject.PARENT))
    self.test_project.handle_create()
    self.projects.create.assert_called_once_with(name='foo', description='', domain='default', enabled=True, parent=None, tags=[])