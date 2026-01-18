import uuid
from openstackclient.tests.functional.network.v2 import common
def test_security_group_list(self):
    cmd_output = self.openstack('security group list', parse_output=True)
    self.assertIn(self.NAME, [sg['Name'] for sg in cmd_output])