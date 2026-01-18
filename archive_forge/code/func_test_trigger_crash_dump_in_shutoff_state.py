import time
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.v2 import shell
def test_trigger_crash_dump_in_shutoff_state(self):
    server = self._create_server()
    self.wait_for_server_os_boot(server.id)
    self.nova('stop %s ' % server.id)
    shell._poll_for_status(self.client.servers.get, server.id, 'active', ['shutoff'])
    output = self.nova('trigger-crash-dump %s ' % server.id, fail_ok=True, merge_stderr=True)
    self.assertIn("ERROR (Conflict): Cannot 'trigger_crash_dump' instance %s while it is in vm_state stopped (HTTP 409) " % server.id, output)