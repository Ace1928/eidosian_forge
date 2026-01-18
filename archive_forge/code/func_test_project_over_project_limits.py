from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_project_over_project_limits(self):
    self.enforcer.enforce('project2', {'widgets': 7})
    self.assertRaises(exception.ProjectOverLimit, self.enforcer.enforce, 'project2', {'widgets': 10})