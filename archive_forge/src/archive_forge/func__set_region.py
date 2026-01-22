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
def _set_region(self, region):
    """
        Return the region to use for listing resources.

        :param  region: A name, region object, None, or 'all'
        :type   region: ``str`` or :class:`GCERegion` or ``None``

        :return:  A region object or None if all regions should be considered
        :rtype:   :class:`GCERegion` or ``None``
        """
    region = region or self.region
    if region == 'all' or region is None:
        return None
    if not hasattr(region, 'name'):
        region = self.ex_get_region(region)
    return region