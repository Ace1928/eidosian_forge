from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_project_over_registered_limit_only(self):
    self.assertRaises(exception.ProjectOverLimit, self.enforcer.enforce, 'project1', {'sprockets': 1, 'widgets': 102})