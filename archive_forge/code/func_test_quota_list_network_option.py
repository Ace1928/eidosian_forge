import uuid
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_quota_list_network_option(self):
    if not self.haz_network:
        self.skipTest('No Network service present')
    self.openstack('quota set --networks 40 ' + self.PROJECT_NAME)
    cmd_output = self.openstack('quota list --network', parse_output=True)
    self.assertIsNotNone(cmd_output)
    self.assertEqual(40, cmd_output[0]['Networks'])