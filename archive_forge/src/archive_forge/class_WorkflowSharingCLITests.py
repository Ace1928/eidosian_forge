from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
class WorkflowSharingCLITests(base_v2.MistralClientTestBase):

    def setUp(self):
        super(WorkflowSharingCLITests, self).setUp()
        self.wf = self.workflow_create(self.wf_def, admin=True)

    def _update_shared_workflow(self, new_status='accepted'):
        member = self.workflow_member_create(self.wf[0]['ID'])
        status = self.get_field_value(member, 'Status')
        self.assertEqual('pending', status)
        cmd_param = '%s workflow --status %s --member-id %s' % (self.wf[0]['ID'], new_status, self.get_project_id('alt_demo'))
        member = self.mistral_alt_user('member-update', params=cmd_param)
        status = self.get_field_value(member, 'Status')
        self.assertEqual(new_status, status)

    def test_list_accepted_shared_workflow(self):
        wfs = self.mistral_alt_user('workflow-list')
        self.assertNotIn(self.wf[0]['ID'], [w['ID'] for w in wfs])
        self._update_shared_workflow(new_status='accepted')
        alt_wfs = self.mistral_alt_user('workflow-list')
        self.assertIn(self.wf[0]['ID'], [w['ID'] for w in alt_wfs])
        self.assertIn(self.get_project_id('admin'), [w['Project ID'] for w in alt_wfs])

    def test_list_rejected_shared_workflow(self):
        self._update_shared_workflow(new_status='rejected')
        alt_wfs = self.mistral_alt_user('workflow-list')
        self.assertNotIn(self.wf[0]['ID'], [w['ID'] for w in alt_wfs])

    def test_create_execution_using_shared_workflow(self):
        self._update_shared_workflow(new_status='accepted')
        execution = self.execution_create(self.wf[0]['ID'], admin=False)
        wf_name = self.get_field_value(execution, 'Workflow name')
        self.assertEqual(self.wf[0]['Name'], wf_name)

    def test_create_contrigger_using_shared_workflow(self):
        self._update_shared_workflow(new_status='accepted')
        trigger = self.cron_trigger_create('test_trigger', self.wf[0]['ID'], '{}', '5 * * * *', admin=False)
        wf_name = self.get_field_value(trigger, 'Workflow')
        self.assertEqual(self.wf[0]['Name'], wf_name)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-delete', params=self.wf[0]['ID'])