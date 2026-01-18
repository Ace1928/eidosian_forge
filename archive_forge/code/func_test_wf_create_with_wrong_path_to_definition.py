import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_wf_create_with_wrong_path_to_definition(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-create', params='wf')