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
def _ex_populate_volume_dict(self):
    """
        Fetch the volume information using disks/aggregatedList
        and store it in _ex_volume_dict.

        return:  ``None``
        """
    aggregated_items = self.connection.request_aggregated_items('disks')
    self._ex_volume_dict = self._build_volume_dict(aggregated_items['items'])
    return None