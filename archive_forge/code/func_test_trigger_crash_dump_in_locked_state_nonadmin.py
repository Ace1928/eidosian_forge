import time
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.v2 import shell
def test_trigger_crash_dump_in_locked_state_nonadmin(self):
    name = self.name_generate()
    server = self.another_nova('boot --flavor %s --image %s --poll %s' % (self.flavor.name, self.image.name, name))
    self.addCleanup(self.another_nova, 'delete', params=name)
    server_id = self._get_value_from_the_table(server, 'id')
    self.wait_for_server_os_boot(server_id)
    self.another_nova('lock %s ' % server_id)
    self.addCleanup(self.another_nova, 'unlock', params=name)
    output = self.another_nova('trigger-crash-dump %s ' % server_id, fail_ok=True, merge_stderr=True)
    self.assertIn('ERROR (Conflict)', output)