import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
def test_set_rules_overwrite_true(self):
    enforcer = policy.Enforcer()
    enforcer.load_rules(True)
    enforcer.set_rules({'test_heat_rule': 1}, True)
    self.assertEqual({'test_heat_rule': 1}, enforcer.enforcer.rules)