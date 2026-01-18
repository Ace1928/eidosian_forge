import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
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