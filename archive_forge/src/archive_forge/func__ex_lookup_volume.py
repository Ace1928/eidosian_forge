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
def _ex_lookup_volume(self, volume_name, zone=None):
    """
        Look up volume by name and zone in volume dict.

        If zone isn't specified or equals 'all', we return the volume
        for the first zone, as determined alphabetically.

        :param    volume_name: The name of the volume.
        :type     volume_name: ``str``

        :keyword  zone: The zone to search for the volume in (set to 'all' to
                        search all zones)
        :type     zone: ``str`` or ``None``

        :return:  A StorageVolume object for the volume.
        :rtype:   :class:`StorageVolume` or raise ``ResourceNotFoundError``.
        """
    if volume_name not in self._ex_volume_dict:
        self._ex_populate_volume_dict()
        if volume_name not in self._ex_volume_dict:
            raise ResourceNotFoundError("Volume name: '{}' not found. Zone: {}".format(volume_name, zone), None, None)
    if zone is None or zone == 'all':
        zone = sorted(self._ex_volume_dict[volume_name])[0]
    volume = self._ex_volume_dict[volume_name].get(zone, None)
    if not volume:
        raise ResourceNotFoundError("Volume '{}' not found for zone {}.".format(volume_name, zone), None, None)
    return self._to_storage_volume(volume)