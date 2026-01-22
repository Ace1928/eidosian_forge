import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def _assert_volume_attach(self, server, volume_id=None, image=''):
    self.assertEqual(self.server_name, server['name'])
    self.assertEqual(image, server['image'])
    self.assertEqual(self.flavor.id, server['flavor']['id'])
    volumes = self.user_cloud.get_volumes(server)
    self.assertEqual(1, len(volumes))
    volume = volumes[0]
    if volume_id:
        self.assertEqual(volume_id, volume['id'])
    else:
        volume_id = volume['id']
    self.assertEqual(1, len(volume['attachments']), 1)
    self.assertEqual(server['id'], volume['attachments'][0]['server_id'])
    return volume_id