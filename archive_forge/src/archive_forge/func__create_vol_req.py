import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _create_vol_req(self, size, name, location=None, snapshot=None, image=None, ex_disk_type='pd-standard'):
    """
        Assemble the request/data for creating a volume.

        Used by create_volume and ex_create_multiple_nodes

        :param  size: Size of volume to create (in GB). Can be None if image
                      or snapshot is supplied.
        :type   size: ``int`` or ``str`` or ``None``

        :param  name: Name of volume to create
        :type   name: ``str``

        :keyword  location: Location (zone) to create the volume in
        :type     location: ``str`` or :class:`GCEZone` or
                            :class:`NodeLocation` or ``None``

        :keyword  snapshot: Snapshot to create image from
        :type     snapshot: :class:`GCESnapshot` or ``str`` or ``None``

        :keyword  image: Image to create disk from.
        :type     image: :class:`GCENodeImage` or ``str`` or ``None``

        :keyword  ex_disk_type: Specify pd-standard (default) or pd-ssd
        :type     ex_disk_type: ``str`` or :class:`GCEDiskType`

        :return:  Tuple containing the request string, the data dictionary and
                  the URL parameters
        :rtype:   ``tuple``
        """
    volume_data = {}
    params = None
    volume_data['name'] = name
    if size:
        volume_data['sizeGb'] = str(size)
    if image:
        if not hasattr(image, 'name'):
            image = self.ex_get_image(image)
        params = {'sourceImage': image.extra['selfLink']}
        volume_data['description'] = 'Image: %s' % image.extra['selfLink']
    if snapshot:
        if not hasattr(snapshot, 'name'):
            if snapshot.startswith('https'):
                snapshot = self._get_components_from_path(snapshot)['name']
            snapshot = self.ex_get_snapshot(snapshot)
        snapshot_link = snapshot.extra['selfLink']
        volume_data['sourceSnapshot'] = snapshot_link
        volume_data['description'] = 'Snapshot: %s' % snapshot_link
    location = location or self.zone
    if not hasattr(location, 'name'):
        location = self.ex_get_zone(location)
    if hasattr(ex_disk_type, 'name'):
        volume_data['type'] = ex_disk_type.extra['selfLink']
    elif ex_disk_type.startswith('https'):
        volume_data['type'] = ex_disk_type
    else:
        volume_data['type'] = 'https://www.googleapis.com/compute/'
        volume_data['type'] += '{}/projects/{}/zones/{}/diskTypes/{}'.format(API_VERSION, self.project, location.name, ex_disk_type)
    request = '/zones/%s/disks' % location.name
    return (request, volume_data, params)