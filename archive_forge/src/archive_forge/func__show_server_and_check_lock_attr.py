import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def _show_server_and_check_lock_attr(self, server, value):
    output = self.nova('show %s' % server.id)
    self.assertEqual(str(value), self._get_value_from_the_table(output, 'locked'))