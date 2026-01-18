import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
def test_resource_enforce_fail(self):
    context = utils.dummy_context(roles=['non-admin'])
    enforcer = policy.ResourceEnforcer()
    res_type = 'OS::Nova::Quota'
    ex = self.assertRaises(exception.Forbidden, enforcer.enforce, context, res_type, None, None, True)
    self.assertIn(res_type, ex.message)