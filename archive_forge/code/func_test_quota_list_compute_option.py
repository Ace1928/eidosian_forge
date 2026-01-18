import uuid
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_quota_list_compute_option(self):
    self.openstack('quota set --instances 30 ' + self.PROJECT_NAME)
    cmd_output = self.openstack('quota list --compute', parse_output=True)
    self.assertIsNotNone(cmd_output)
    self.assertEqual(30, cmd_output[0]['Instances'])