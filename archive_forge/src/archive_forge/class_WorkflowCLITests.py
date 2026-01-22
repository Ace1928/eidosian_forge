import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class WorkflowCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with workflows."""

    @classmethod
    def setUpClass(cls):
        super(WorkflowCLITests, cls).setUpClass()

    def test_workflow_create_delete(self):
        init_wfs = self.mistral_admin('workflow-create', params=self.wf_def)
        wf_names = [wf['Name'] for wf in init_wfs]
        self.assertTableStruct(init_wfs, ['Name', 'Created at', 'Updated at'])
        wfs = self.mistral_admin('workflow-list')
        self.assertIn(wf_names[0], [workflow['Name'] for workflow in wfs])
        for wf_name in wf_names:
            self.mistral_admin('workflow-delete', params=wf_name)
        wfs = self.mistral_admin('workflow-list')
        for wf in wf_names:
            self.assertNotIn(wf, [workflow['Name'] for workflow in wfs])

    def test_workflow_within_namespace_create_delete(self):
        params = self.wf_def + ' --namespace abcdef'
        init_wfs = self.mistral_admin('workflow-create', params=params)
        wf_names = [wf['Name'] for wf in init_wfs]
        self.assertTableStruct(init_wfs, ['Name', 'Created at', 'Updated at'])
        wfs = self.mistral_admin('workflow-list')
        self.assertIn(wf_names[0], [workflow['Name'] for workflow in wfs])
        for wf_name in wf_names:
            self.mistral_admin('workflow-delete', params=wf_name + ' --namespace abcdef')
        wfs = self.mistral_admin('workflow-list')
        for wf in wf_names:
            self.assertNotIn(wf, [workflow['Name'] for workflow in wfs])
        init_wfs = self.mistral_admin('workflow-create', params=params)
        wf_ids = [wf['ID'] for wf in init_wfs]
        for wf_id in wf_ids:
            self.mistral_admin('workflow-delete', params=wf_id)
        for wf in wf_names:
            self.assertNotIn(wf, [workflow['Name'] for workflow in wfs])

    def test_create_wf_with_tags(self):
        init_wfs = self.workflow_create(self.wf_def)
        wf_name = init_wfs[1]['Name']
        self.assertTableStruct(init_wfs, ['Name', 'Created at', 'Updated at', 'Tags'])
        created_wf_info = self.get_item_info(get_from=init_wfs, get_by='Name', value=wf_name)
        self.assertEqual('tag', created_wf_info['Tags'])

    def test_workflow_update(self):
        wf = self.workflow_create(self.wf_def)
        wf_name = wf[0]['Name']
        wf_id = wf[0]['ID']
        created_wf_info = self.get_item_info(get_from=wf, get_by='Name', value=wf_name)
        upd_wf = self.mistral_admin('workflow-update', params='{0}'.format(self.wf_def))
        self.assertTableStruct(upd_wf, ['Name', 'Created at', 'Updated at'])
        updated_wf_info = self.get_item_info(get_from=upd_wf, get_by='Name', value=wf_name)
        self.assertEqual(wf_name, upd_wf[0]['Name'])
        self.assertEqual(created_wf_info['Created at'].split('.')[0], updated_wf_info['Created at'])
        self.assertEqual(created_wf_info['Updated at'], updated_wf_info['Updated at'])
        upd_wf = self.mistral_admin('workflow-update', params='{0}'.format(self.wf_with_delay_def))
        self.assertTableStruct(upd_wf, ['Name', 'Created at', 'Updated at'])
        updated_wf_info = self.get_item_info(get_from=upd_wf, get_by='Name', value=wf_name)
        self.assertEqual(wf_name, upd_wf[0]['Name'])
        self.assertEqual(created_wf_info['Created at'].split('.')[0], updated_wf_info['Created at'])
        self.assertNotEqual(created_wf_info['Updated at'], updated_wf_info['Updated at'])
        upd_wf = self.mistral_admin('workflow-update', params='{0} --id {1}'.format(self.wf_with_delay_def, wf_id))
        self.assertTableStruct(upd_wf, ['Name', 'Created at', 'Updated at'])
        updated_wf_info = self.get_item_info(get_from=upd_wf, get_by='ID', value=wf_id)
        self.assertEqual(wf_name, upd_wf[0]['Name'])
        self.assertEqual(created_wf_info['Created at'].split('.')[0], updated_wf_info['Created at'])
        self.assertNotEqual(created_wf_info['Updated at'], updated_wf_info['Updated at'])

    def test_workflow_update_within_namespace(self):
        namespace = 'abc'
        wf = self.workflow_create(self.wf_def, namespace=namespace)
        wf_name = wf[0]['Name']
        wf_id = wf[0]['ID']
        wf_namespace = wf[0]['Namespace']
        created_wf_info = self.get_item_info(get_from=wf, get_by='Name', value=wf_name)
        upd_wf = self.mistral_admin('workflow-update', params='{0} --namespace {1}'.format(self.wf_def, namespace))
        self.assertTableStruct(upd_wf, ['Name', 'Created at', 'Updated at'])
        updated_wf_info = self.get_item_info(get_from=upd_wf, get_by='Name', value=wf_name)
        self.assertEqual(wf_name, upd_wf[0]['Name'])
        self.assertEqual(namespace, wf_namespace)
        self.assertEqual(wf_namespace, upd_wf[0]['Namespace'])
        self.assertEqual(created_wf_info['Created at'].split('.')[0], updated_wf_info['Created at'])
        self.assertEqual(created_wf_info['Updated at'], updated_wf_info['Updated at'])
        upd_wf = self.mistral_admin('workflow-update', params='{0} --namespace {1}'.format(self.wf_with_delay_def, namespace))
        self.assertTableStruct(upd_wf, ['Name', 'Created at', 'Updated at'])
        updated_wf_info = self.get_item_info(get_from=upd_wf, get_by='Name', value=wf_name)
        self.assertEqual(wf_name, upd_wf[0]['Name'])
        self.assertEqual(created_wf_info['Created at'].split('.')[0], updated_wf_info['Created at'])
        self.assertNotEqual(created_wf_info['Updated at'], updated_wf_info['Updated at'])
        upd_wf = self.mistral_admin('workflow-update', params='{0} --id {1}'.format(self.wf_with_delay_def, wf_id))
        self.assertTableStruct(upd_wf, ['Name', 'Created at', 'Updated at'])
        updated_wf_info = self.get_item_info(get_from=upd_wf, get_by='ID', value=wf_id)
        self.assertEqual(wf_name, upd_wf[0]['Name'])
        self.assertEqual(created_wf_info['Created at'].split('.')[0], updated_wf_info['Created at'])
        self.assertNotEqual(created_wf_info['Updated at'], updated_wf_info['Updated at'])

    def test_workflow_update_truncate_input(self):
        input_value = 'very_long_input_parameter_name_that_should_be_truncated'
        wf_def = '\n        version: "2.0"\n        workflow1:\n          input:\n            - {0}\n          tasks:\n            task1:\n              action: std.noop\n        '.format(input_value)
        self.create_file('wf.yaml', wf_def)
        self.workflow_create('wf.yaml')
        updated_wf = self.mistral_admin('workflow-update', params='wf.yaml')
        updated_wf_info = self.get_item_info(get_from=updated_wf, get_by='Name', value='workflow1')
        self.assertEqual(updated_wf_info['Input'][:-3], input_value[:25])

    def test_workflow_get(self):
        created = self.workflow_create(self.wf_def)
        wf_name = created[0]['Name']
        fetched = self.mistral_admin('workflow-get', params=wf_name)
        fetched_wf_name = self.get_field_value(fetched, 'Name')
        self.assertEqual(wf_name, fetched_wf_name)

    def test_workflow_get_with_id(self):
        created = self.workflow_create(self.wf_def)
        wf_name = created[0]['Name']
        wf_id = created[0]['ID']
        fetched = self.mistral_admin('workflow-get', params=wf_id)
        fetched_wf_name = self.get_field_value(fetched, 'Name')
        self.assertEqual(wf_name, fetched_wf_name)

    def test_workflow_get_definition(self):
        wf = self.workflow_create(self.wf_def)
        wf_name = wf[0]['Name']
        definition = self.mistral_admin('workflow-get-definition', params=wf_name)
        self.assertNotIn('404 Not Found', definition)

    def test_workflow_validate_with_valid_def(self):
        wf = self.mistral_admin('workflow-validate', params=self.wf_def)
        wf_valid = self.get_field_value(wf, 'Valid')
        wf_error = self.get_field_value(wf, 'Error')
        self.assertEqual('True', wf_valid)
        self.assertEqual('None', wf_error)

    def test_workflow_validate_with_invalid_def(self):
        self.create_file('wf.yaml', 'name: wf\n')
        wf = self.mistral_admin('workflow-validate', params='wf.yaml')
        wf_valid = self.get_field_value(wf, 'Valid')
        wf_error = self.get_field_value(wf, 'Error')
        self.assertEqual('False', wf_valid)
        self.assertNotEqual('None', wf_error)

    def test_workflow_list_with_filter(self):
        self.workflow_create(self.wf_def)
        workflows = self.parser.listing(self.mistral('workflow-list'))
        self.assertTableStruct(workflows, ['ID', 'Name', 'Tags', 'Input', 'Scope', 'Created at', 'Updated at'])
        self.assertGreaterEqual(len(workflows), 2)
        workflows = self.parser.listing(self.mistral('workflow-list', params='--filter name=eq:wf1'))
        self.assertTableStruct(workflows, ['ID', 'Name', 'Tags', 'Input', 'Scope', 'Created at', 'Updated at'])
        self.assertEqual(1, len(workflows))
        self.assertEqual('wf1', workflows[0]['Name'])