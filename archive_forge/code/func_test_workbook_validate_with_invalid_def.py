import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbook_validate_with_invalid_def(self):
    self.create_file('wb.yaml', 'name: wb\n')
    wb = self.mistral_admin('workbook-validate', params='wb.yaml')
    wb_valid = self.get_field_value(wb, 'Valid')
    wb_error = self.get_field_value(wb, 'Error')
    self.assertEqual('False', wb_valid)
    self.assertNotEqual('None', wb_error)