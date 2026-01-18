from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
def test_wait_secs_create(self):
    stk = utils.parse_stack(self.simple_template)
    action = status.ResourceStatus.CREATE
    self.assertEqual(3, stk['constant']._wait_secs(action))
    variable = stk['variable']._wait_secs(action)
    self.assertGreaterEqual(variable, 1.6)
    self.assertLessEqual(variable, 5.8)
    self.assertNotEqual(variable, stk['variable']._wait_secs(action))
    self.assertEqual(0, stk['variable_prod']._wait_secs(action))