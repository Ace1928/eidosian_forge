from heat_integrationtests.functional import functional_base
def test_replaced_resource(self):
    self.stack_identifier = self.stack_create(template=test_template_one_resource)
    new_template = {'heat_template_version': '2013-05-23', 'description': 'Test template to create one instance.', 'resources': {'test1': {'type': 'OS::Heat::TestResource', 'properties': {'update_replace': True}}}}
    result = self.preview_update_stack(self.stack_identifier, new_template)
    changes = result['resource_changes']
    replaced = changes['replaced'][0]['resource_name']
    self.assertEqual('test1', replaced)
    self.assert_empty_sections(changes, ['added', 'unchanged', 'updated', 'deleted'])