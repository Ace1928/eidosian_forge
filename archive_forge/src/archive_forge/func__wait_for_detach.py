import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def _wait_for_detach(self, volume_id):
    for count in utils.iterate_timeout(60, 'Timeout waiting for volume {volume_id} to detach'.format(volume_id=volume_id)):
        volume = self.user_cloud.get_volume(volume_id)
        if volume.status in ('available', 'error', 'error_restoring', 'error_extending'):
            return