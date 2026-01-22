import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def _test_server_boot_with_bdm_snapshot(self, use_legacy):
    """Test server create from image with bdm snapshot, server delete"""
    volume_wait_for = volume_common.BaseVolumeTests.wait_for_status
    volume_wait_for_delete = volume_common.BaseVolumeTests.wait_for_delete
    empty_volume_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 1 ' + empty_volume_name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.addCleanup(self.openstack, 'volume delete ' + empty_volume_name)
    self.assertEqual(empty_volume_name, cmd_output['name'])
    volume_wait_for('volume', empty_volume_name, 'available')
    empty_snapshot_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume snapshot create ' + '--volume ' + empty_volume_name + ' ' + empty_snapshot_name, parse_output=True)
    empty_snapshot_id = cmd_output['id']
    self.assertIsNotNone(empty_snapshot_id)
    self.addCleanup(volume_wait_for_delete, 'volume snapshot', empty_snapshot_name)
    self.addCleanup(self.openstack, 'volume snapshot delete ' + empty_snapshot_name)
    self.assertEqual(empty_snapshot_name, cmd_output['name'])
    volume_wait_for('volume snapshot', empty_snapshot_name, 'available')
    if use_legacy:
        bdm_arg = f'--block-device-mapping vdb={empty_snapshot_name}:snapshot:1:true'
    else:
        bdm_arg = f'--block-device device_name=vdb,uuid={empty_snapshot_id},source_type=snapshot,volume_size=1,delete_on_termination=true,boot_index=1'
    server_name = uuid.uuid4().hex
    server = self.openstack('server create ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + bdm_arg + ' ' + self.network_arg + ' ' + '--wait ' + server_name, parse_output=True)
    self.assertIsNotNone(server['id'])
    self.assertEqual(server_name, server['name'])
    self.wait_for_status(server_name, 'ACTIVE')
    cmd_output = self.openstack('server show ' + server_name, parse_output=True)
    volumes_attached = cmd_output['volumes_attached']
    self.assertIsNotNone(volumes_attached)
    attached_volume_id = volumes_attached[0]['id']
    cmd_output = self.openstack('volume show ' + attached_volume_id, parse_output=True)
    attachments = cmd_output['attachments']
    self.assertEqual(1, len(attachments))
    self.assertEqual(server['id'], attachments[0]['server_id'])
    self.assertEqual('in-use', cmd_output['status'])
    self.openstack('server delete --wait ' + server_name)
    cmd_output = self.openstack('volume list', parse_output=True)
    target_volume = [each_volume for each_volume in cmd_output if each_volume['ID'] == attached_volume_id]
    if target_volume:
        self.assertEqual('deleting', target_volume[0]['Status'])
    else:
        pass