import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_list_with_marker_and_deleted(self):
    """Test server list with deleted and marker"""
    cmd_output = self.server_create(cleanup=False)
    name1 = cmd_output['name']
    cmd_output = self.server_create(cleanup=False)
    name2 = cmd_output['name']
    id2 = cmd_output['id']
    self.wait_for_status(name1, 'ACTIVE')
    self.wait_for_status(name2, 'ACTIVE')
    cmd_output = self.openstack('server list --marker ' + id2, parse_output=True)
    col_name = [x['Name'] for x in cmd_output]
    self.assertIn(name1, col_name)
    cmd_output = self.openstack('server list --marker ' + name2, parse_output=True)
    col_name = [x['Name'] for x in cmd_output]
    self.assertIn(name1, col_name)
    self.openstack('server delete --wait ' + name1)
    self.openstack('server delete --wait ' + name2)
    cmd_output = self.openstack('server list --deleted --marker ' + id2, parse_output=True)
    col_name = [x['Name'] for x in cmd_output]
    self.assertIn(name1, col_name)
    try:
        cmd_output = self.openstack('server list --deleted --marker ' + name2, parse_output=True)
    except exceptions.CommandFailed as e:
        self.assertIn('marker [%s] not found' % name2, e.stderr.decode('utf-8'))