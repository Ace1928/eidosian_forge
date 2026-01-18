import uuid
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_quota_list_volume_option(self):
    self.openstack('quota set --volumes 20 ' + self.PROJECT_NAME)
    cmd_output = self.openstack('quota list --volume', parse_output=True)
    self.assertIsNotNone(cmd_output)
    self.assertEqual(20, cmd_output[0]['Volumes'])