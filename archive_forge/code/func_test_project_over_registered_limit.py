from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_project_over_registered_limit(self):
    self.enforcer.enforce('project2', {'sprockets': 1})
    self.assertRaises(exception.ProjectOverLimit, self.enforcer.enforce, 'project2', {'sprockets': 50})