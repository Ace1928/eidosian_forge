import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbook_create_delete(self):
    wb = self.mistral_admin('workbook-create', params=self.wb_def)
    wb_name = self.get_field_value(wb, 'Name')
    self.assertTableStruct(wb, ['Field', 'Value'])
    wbs = self.mistral_admin('workbook-list')
    self.assertIn(wb_name, [w['Name'] for w in wbs])
    wbs = self.mistral_admin('workbook-list')
    self.assertIn(wb_name, [w['Name'] for w in wbs])
    self.mistral_admin('workbook-delete', params=wb_name)
    wbs = self.mistral_admin('workbook-list')
    self.assertNotIn(wb_name, [w['Name'] for w in wbs])