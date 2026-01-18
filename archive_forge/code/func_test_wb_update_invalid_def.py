import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_wb_update_invalid_def(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-update', params=self.wf_def)