import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_tr_create_nonexistent_wf(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --pattern "* * * * *"')