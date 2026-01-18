from unittest import mock
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.keystone import project
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_project_handle_update_only_enabled(self):
    self.test_project.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {project.KeystoneProject.ENABLED: False}
    self.test_project.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.projects.update.assert_called_once_with(project=self.test_project.resource_id, name=None, description=None, enabled=prop_diff[project.KeystoneProject.ENABLED], domain='default', tags=['label', 'insignia'])