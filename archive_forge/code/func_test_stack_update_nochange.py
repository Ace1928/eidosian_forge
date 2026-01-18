import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_nochange(self):
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_no_change'})
    stack_identifier = self.stack_create(template=template)
    expected_resources = {'test1': 'OS::Heat::TestResource'}
    self.assertEqual(expected_resources, self.list_resources(stack_identifier))
    self.update_stack(stack_identifier, template)
    self.assertEqual(expected_resources, self.list_resources(stack_identifier))