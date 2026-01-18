from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_wf_isolation(self):
    wf = self.workflow_create(self.wf_def)
    wfs = self.mistral_admin('workflow-list')
    self.assertIn(wf[0]['Name'], [w['Name'] for w in wfs])
    alt_wfs = self.mistral_alt_user('workflow-list')
    self.assertNotIn(wf[0]['Name'], [w['Name'] for w in alt_wfs])