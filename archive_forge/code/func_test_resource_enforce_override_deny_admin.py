import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
def test_resource_enforce_override_deny_admin(self):
    context = utils.dummy_context(roles=['admin'])
    enforcer = policy.ResourceEnforcer(policy_file=self.get_policy_file('resources.json'))
    res_type = 'OS::Cinder::Quota'
    ex = self.assertRaises(exception.Forbidden, enforcer.enforce, context, res_type, None, None, True)
    self.assertIn(res_type, ex.message)