import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbook_get_definition(self):
    wb = self.workbook_create(self.wb_def)
    wb_name = self.get_field_value(wb, 'Name')
    definition = self.mistral_admin('workbook-get-definition', params=wb_name)
    self.assertNotIn('404 Not Found', definition)