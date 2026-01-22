from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
class TestDelay(common.HeatTestCase):
    simple_template = template_format.parse("\nheat_template_version: '2016-10-14'\nresources:\n  constant:\n    type: OS::Heat::Delay\n    properties:\n      min_wait: 3\n  variable:\n    type: OS::Heat::Delay\n    properties:\n      min_wait: 1.6\n      max_jitter: 4.2\n      actions:\n        - CREATE\n        - DELETE\n  variable_prod:\n    type: OS::Heat::Delay\n    properties:\n      min_wait: 2\n      max_jitter: 666\n      jitter_multiplier: 0.1\n      actions:\n        - DELETE\n")

    def test_delay_params(self):
        stk = utils.parse_stack(self.simple_template)
        self.assertEqual((3, 0), stk['constant']._delay_parameters())
        self.assertEqual((1.6, 4.2), stk['variable']._delay_parameters())
        min_wait, max_jitter = stk['variable_prod']._delay_parameters()
        self.assertEqual(2, min_wait)
        self.assertAlmostEqual(66.6, max_jitter)

    def test_wait_secs_create(self):
        stk = utils.parse_stack(self.simple_template)
        action = status.ResourceStatus.CREATE
        self.assertEqual(3, stk['constant']._wait_secs(action))
        variable = stk['variable']._wait_secs(action)
        self.assertGreaterEqual(variable, 1.6)
        self.assertLessEqual(variable, 5.8)
        self.assertNotEqual(variable, stk['variable']._wait_secs(action))
        self.assertEqual(0, stk['variable_prod']._wait_secs(action))

    def test_wait_secs_delete(self):
        stk = utils.parse_stack(self.simple_template)
        action = status.ResourceStatus.DELETE
        self.assertEqual(0, stk['constant']._wait_secs(action))
        variable = stk['variable']._wait_secs(action)
        self.assertGreaterEqual(variable, 1.6)
        self.assertLessEqual(variable, 5.8)
        self.assertNotEqual(variable, stk['variable']._wait_secs(action))
        variable_prod = stk['variable_prod']._wait_secs(action)
        self.assertGreaterEqual(variable_prod, 2.0)
        self.assertLessEqual(variable_prod, 68.6)
        self.assertNotEqual(variable_prod, stk['variable_prod']._wait_secs(action))

    def test_wait_secs_update(self):
        stk = utils.parse_stack(self.simple_template)
        action = status.ResourceStatus.UPDATE
        self.assertEqual(0, stk['constant']._wait_secs(action))
        self.assertEqual(0, stk['variable']._wait_secs(action))
        self.assertEqual(0, stk['variable_prod']._wait_secs(action))

    def test_validate_success(self):
        stk = utils.parse_stack(self.simple_template)
        for res in stk.resources.values():
            self.assertIsNone(res.validate())

    def test_validate_failure(self):
        stk = utils.parse_stack(self.simple_template)
        stk.timeout_mins = 1
        self.assertRaises(exception.StackValidationFailed, stk['variable_prod'].validate)