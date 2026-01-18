import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_list(self):
    """Test server list"""
    cmd_output = self.server_create()
    name1 = cmd_output['name']
    cmd_output = self.server_create()
    name2 = cmd_output['name']
    self.wait_for_status(name1, 'ACTIVE')
    self.wait_for_status(name2, 'ACTIVE')
    cmd_output = self.openstack('server list', parse_output=True)
    col_name = [x['Name'] for x in cmd_output]
    self.assertIn(name1, col_name)
    self.assertIn(name2, col_name)
    raw_output = self.openstack('server pause ' + name2)
    self.assertEqual('', raw_output)
    self.wait_for_status(name2, 'PAUSED')
    cmd_output = self.openstack('server list ' + '--status ACTIVE', parse_output=True)
    col_name = [x['Name'] for x in cmd_output]
    self.assertIn(name1, col_name)
    self.assertNotIn(name2, col_name)
    cmd_output = self.openstack('server list ' + '--status PAUSED', parse_output=True)
    col_name = [x['Name'] for x in cmd_output]
    self.assertNotIn(name1, col_name)
    self.assertIn(name2, col_name)