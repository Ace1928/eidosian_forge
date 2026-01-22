from heat_integrationtests.functional import functional_base
class UpdatePreviewStackTest(UpdatePreviewBase):

    def test_add_resource(self):
        self.stack_identifier = self.stack_create(template=test_template_one_resource)
        result = self.preview_update_stack(self.stack_identifier, test_template_two_resource)
        changes = result['resource_changes']
        unchanged = changes['unchanged'][0]['resource_name']
        self.assertEqual('test1', unchanged)
        added = changes['added'][0]['resource_name']
        self.assertEqual('test2', added)
        self.assert_empty_sections(changes, ['updated', 'replaced', 'deleted'])

    def test_no_change(self):
        self.stack_identifier = self.stack_create(template=test_template_one_resource)
        result = self.preview_update_stack(self.stack_identifier, test_template_one_resource)
        changes = result['resource_changes']
        unchanged = changes['unchanged'][0]['resource_name']
        self.assertEqual('test1', unchanged)
        self.assert_empty_sections(changes, ['updated', 'replaced', 'deleted', 'added'])

    def test_update_resource(self):
        self.stack_identifier = self.stack_create(template=test_template_one_resource)
        test_template_updated_resource = {'heat_template_version': '2013-05-23', 'description': 'Test template to create one instance.', 'resources': {'test1': {'type': 'OS::Heat::TestResource', 'properties': {'value': 'Test1 foo', 'fail': False, 'update_replace': False, 'wait_secs': 0}}}}
        result = self.preview_update_stack(self.stack_identifier, test_template_updated_resource)
        changes = result['resource_changes']
        updated = changes['updated'][0]['resource_name']
        self.assertEqual('test1', updated)
        self.assert_empty_sections(changes, ['added', 'unchanged', 'replaced', 'deleted'])

    def test_replaced_resource(self):
        self.stack_identifier = self.stack_create(template=test_template_one_resource)
        new_template = {'heat_template_version': '2013-05-23', 'description': 'Test template to create one instance.', 'resources': {'test1': {'type': 'OS::Heat::TestResource', 'properties': {'update_replace': True}}}}
        result = self.preview_update_stack(self.stack_identifier, new_template)
        changes = result['resource_changes']
        replaced = changes['replaced'][0]['resource_name']
        self.assertEqual('test1', replaced)
        self.assert_empty_sections(changes, ['added', 'unchanged', 'updated', 'deleted'])

    def test_delete_resource(self):
        self.stack_identifier = self.stack_create(template=test_template_two_resource)
        result = self.preview_update_stack(self.stack_identifier, test_template_one_resource)
        changes = result['resource_changes']
        unchanged = changes['unchanged'][0]['resource_name']
        self.assertEqual('test1', unchanged)
        deleted = changes['deleted'][0]['resource_name']
        self.assertEqual('test2', deleted)
        self.assert_empty_sections(changes, ['updated', 'replaced', 'added'])