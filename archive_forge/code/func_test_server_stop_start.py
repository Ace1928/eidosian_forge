import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_stop_start(self):
    """Test server stop, start"""
    server_name = uuid.uuid4().hex
    cmd_output = self.openstack('server create ' + '--network private ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + '--wait ' + server_name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(server_name, cmd_output['name'])
    self.addCleanup(self.openstack, 'server delete --wait ' + server_name)
    server_id = cmd_output['id']
    cmd_output = self.openstack('server stop ' + server_name)
    self.assertEqual('', cmd_output)
    self.wait_for_status(server_id, 'SHUTOFF')
    cmd_output = self.openstack('server start ' + server_name)
    self.assertEqual('', cmd_output)
    self.wait_for_status(server_id, 'ACTIVE')