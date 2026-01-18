import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_tr_create_without_pattern(self):
    wf = self.workflow_create(self.wf_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr %s {}' % wf[0]['Name'])