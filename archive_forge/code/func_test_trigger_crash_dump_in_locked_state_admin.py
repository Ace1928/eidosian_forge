import time
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.v2 import shell
def test_trigger_crash_dump_in_locked_state_admin(self):
    server = self._create_server()
    self.wait_for_server_os_boot(server.id)
    self.nova('lock %s ' % server.id)
    self.nova('trigger-crash-dump %s ' % server.id)
    self._assert_nmi(server.id)