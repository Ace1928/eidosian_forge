import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_unicode_tag_correctly_displayed(self):
    """Regression test for bug #1669683.

        List and dict fields with unicode cannot be correctly
        displayed.

        Ensure that once we fix this it doesn't regress.
        """
    uuid = self._boot_server_with_tags(tags=['中文标签'])
    output = self.nova('show %s' % uuid)
    self.assertEqual('["中文标签"]', self._get_value_from_the_table(output, 'tags'))