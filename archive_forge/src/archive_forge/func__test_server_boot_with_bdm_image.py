import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def _test_server_boot_with_bdm_image(self, use_legacy):
    if use_legacy:
        bdm_arg = f'--block-device-mapping vdb={self.image_name}:image:1:true '
    else:
        cmd_output = self.openstack('image show ' + self.image_name, parse_output=True)
        image_id = cmd_output['id']
        bdm_arg = f'--block-device device_name=vdb,uuid={image_id},source_type=image,volume_size=1,delete_on_termination=true,boot_index=1'
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