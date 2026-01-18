import time
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.v2 import shell
def test_trigger_crash_dump_in_resized_state(self):
    server = self._create_server()
    self.wait_for_server_os_boot(server.id)
    self.nova('resize %s %s' % (server.id, 'm1.small'))
    shell._poll_for_status(self.client.servers.get, server.id, 'active', ['verify_resize'])
    self.nova('trigger-crash-dump %s ' % server.id)
    self._assert_nmi(server.id)