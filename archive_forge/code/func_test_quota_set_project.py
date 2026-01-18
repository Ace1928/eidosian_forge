import uuid
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_quota_set_project(self):
    """Test quota set, show"""
    network_option = ''
    if self.haz_network:
        network_option = '--routers 21 '
    self.openstack('quota set --cores 31 --backups 41 ' + network_option + self.PROJECT_NAME)
    cmd_output = self.openstack('quota show ' + self.PROJECT_NAME, parse_output=True)
    cmd_output = {x['Resource']: x['Limit'] for x in cmd_output}
    self.assertIsNotNone(cmd_output)
    self.assertEqual(31, cmd_output['cores'])
    self.assertEqual(41, cmd_output['backups'])
    if self.haz_network:
        self.assertEqual(21, cmd_output['routers'])
    cmd_output = self.openstack('quota show --default', parse_output=True)
    self.assertIsNotNone(cmd_output)
    cmd_output = {x['Resource']: x['Limit'] for x in cmd_output}
    self.assertTrue(cmd_output['cores'] >= 0)
    self.assertTrue(cmd_output['backups'] >= 0)
    if self.haz_network:
        self.assertTrue(cmd_output['routers'] >= 0)