import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
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