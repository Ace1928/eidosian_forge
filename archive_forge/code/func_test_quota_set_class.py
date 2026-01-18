import uuid
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_quota_set_class(self):
    self.openstack('quota set --key-pairs 33 --snapshots 43 ' + '--class default')
    cmd_output = self.openstack('quota show --class default', parse_output=True)
    self.assertIsNotNone(cmd_output)
    cmd_output = {x['Resource']: x['Limit'] for x in cmd_output}
    self.assertEqual(33, cmd_output['key-pairs'])
    self.assertEqual(43, cmd_output['snapshots'])
    cmd_output = self.openstack('quota show --class', parse_output=True)
    self.assertIsNotNone(cmd_output)
    cmd_output = {x['Resource']: x['Limit'] for x in cmd_output}
    self.assertTrue(cmd_output['key-pairs'] >= 0)
    self.assertTrue(cmd_output['snapshots'] >= 0)