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
def _get_region_from_zone(self, zone):
    """
        Return the Region object that contains the given Zone object.

        :param  zone: Zone object
        :type   zone: :class:`GCEZone`

        :return:  Region object that contains the zone
        :rtype:   :class:`GCERegion`
        """
    for region in self.region_list:
        zones = [z.name for z in region.zones]
        if zone.name in zones:
            return region