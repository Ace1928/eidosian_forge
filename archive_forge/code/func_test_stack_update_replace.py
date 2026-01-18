import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_replace(self):
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_replace'})
    stack_identifier = self.stack_create(template=template)
    expected_resources = {'test1': 'OS::Heat::TestResource'}
    self.assertEqual(expected_resources, self.list_resources(stack_identifier))
    resource = self.client.resources.list(stack_identifier)
    initial_phy_id = resource[0].physical_resource_id
    tmpl_update = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_in_place_update', 'update_replace': True})
    self.update_stack(stack_identifier, tmpl_update)
    resource = self.client.resources.list(stack_identifier)
    self.assertNotEqual(initial_phy_id, resource[0].physical_resource_id)