import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_signal_with_policy_update(self):
    """Prove that an updated policy is used in the next signal."""
    stack_identifier = self.stack_create(template=self.template, files=self.files, environment=self.env)
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
    self.client.resources.signal(stack_identifier, 'ScaleUpPolicy')
    self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 3))
    new_template = self.template.replace('"ScalingAdjustment": "1"', '"ScalingAdjustment": "2"').replace('"DesiredCapacity" : {"Ref": "size"},', '')
    self.update_stack(stack_identifier, template=new_template, environment=self.env, files=self.files)
    self.client.resources.signal(stack_identifier, 'ScaleUpPolicy')
    self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 5))