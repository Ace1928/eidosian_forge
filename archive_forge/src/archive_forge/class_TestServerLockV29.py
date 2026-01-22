import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
class TestServerLockV29(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.9'

    def _show_server_and_check_lock_attr(self, server, value):
        output = self.nova('show %s' % server.id)
        self.assertEqual(str(value), self._get_value_from_the_table(output, 'locked'))

    def test_attribute_presented(self):
        server = self._create_server()
        self._show_server_and_check_lock_attr(server, False)
        self.nova('lock %s' % server.id)
        self._show_server_and_check_lock_attr(server, True)
        self.nova('unlock %s' % server.id)
        self._show_server_and_check_lock_attr(server, False)