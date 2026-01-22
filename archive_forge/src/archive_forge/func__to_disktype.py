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
def _to_disktype(self, disktype):
    """
        Return a DiskType object from the JSON-response dictionary.

        :param  disktype: The dictionary describing the disktype.
        :type   disktype: ``dict``

        :return: DiskType object
        :rtype: :class:`GCEDiskType`
        """
    extra = {}
    zone = self.ex_get_zone(disktype['zone'])
    extra['selfLink'] = disktype.get('selfLink')
    extra['creationTimestamp'] = disktype.get('creationTimestamp')
    extra['description'] = disktype.get('description')
    extra['valid_disk_size'] = disktype.get('validDiskSize')
    extra['default_disk_size_gb'] = disktype.get('defaultDiskSizeGb')
    type_id = '{}:{}'.format(zone.name, disktype['name'])
    return GCEDiskType(id=type_id, name=disktype['name'], zone=zone, driver=self, extra=extra)