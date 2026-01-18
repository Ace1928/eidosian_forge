import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
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