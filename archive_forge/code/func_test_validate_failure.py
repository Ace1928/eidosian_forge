from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
def test_validate_failure(self):
    stk = utils.parse_stack(self.simple_template)
    stk.timeout_mins = 1
    self.assertRaises(exception.StackValidationFailed, stk['variable_prod'].validate)