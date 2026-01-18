import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_scaling_meta_update(self):
    """Use heatclient to signal the up and down policy.

        Then confirm that the metadata in the custom_lb is updated each
        time.
        """
    stack_identifier = self.stack_create(template=self.template, files=self.files, environment=self.env)
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
    self.client.resources.signal(stack_identifier, 'ScaleUpPolicy')
    self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 3))
    self.client.resources.signal(stack_identifier, 'ScaleDownPolicy')
    self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 1))