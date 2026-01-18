import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_boot_from_volume(self):
    """Test server create from volume, server delete"""
    volume_wait_for = volume_common.BaseVolumeTests.wait_for_status
    cmd_output = self.openstack('image show ' + self.image_name, parse_output=True)
    try:
        image_size = cmd_output['min_disk']
        if image_size < 1:
            image_size = 1
    except ValueError:
        image_size = 1
    volume_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--image ' + self.image_name + ' ' + '--size ' + str(image_size) + ' ' + volume_name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.addCleanup(self.openstack, 'volume delete ' + volume_name)
    self.assertEqual(volume_name, cmd_output['name'])
    volume_wait_for('volume', volume_name, 'available')
    empty_volume_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size ' + str(image_size) + ' ' + empty_volume_name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.addCleanup(self.openstack, 'volume delete ' + empty_volume_name)
    self.assertEqual(empty_volume_name, cmd_output['name'])
    volume_wait_for('volume', empty_volume_name, 'available')
    server_name = uuid.uuid4().hex
    server = self.openstack('server create ' + '--flavor ' + self.flavor_name + ' ' + '--volume ' + volume_name + ' ' + '--block-device-mapping vdb=' + empty_volume_name + ' ' + self.network_arg + ' ' + '--wait ' + server_name, parse_output=True)
    self.assertIsNotNone(server['id'])
    self.addCleanup(self.openstack, 'server delete --wait ' + server_name)
    self.assertEqual(server_name, server['name'])
    self.assertEqual(v2_server.IMAGE_STRING_FOR_BFV, server['image'])
    servers = self.openstack('server list', parse_output=True)
    self.assertEqual(v2_server.IMAGE_STRING_FOR_BFV, servers[0]['Image'])
    cmd_output = self.openstack('volume show ' + volume_name, parse_output=True)
    attachments = cmd_output['attachments']
    self.assertEqual(1, len(attachments))
    self.assertEqual(server['id'], attachments[0]['server_id'])
    self.assertEqual('in-use', cmd_output['status'])
    cmd_output = self.openstack('volume show ' + empty_volume_name, parse_output=True)
    attachments = cmd_output['attachments']
    self.assertEqual(1, len(attachments))
    self.assertEqual(server['id'], attachments[0]['server_id'])
    self.assertEqual('in-use', cmd_output['status'])