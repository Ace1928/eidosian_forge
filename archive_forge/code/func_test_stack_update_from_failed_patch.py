import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_from_failed_patch(self):
    """Test PATCH update from a failed state."""
    stack_identifier = self.stack_create(template='heat_template_version: 2014-10-16')
    self.update_stack(stack_identifier, template=self.fail_param_template, parameters={'do_fail': True}, expected_status='UPDATE_FAILED')
    self.update_stack(stack_identifier, parameters={'do_fail': False}, existing=True)
    self.assertEqual({u'aresource': u'OS::Heat::TestResource'}, self.list_resources(stack_identifier))