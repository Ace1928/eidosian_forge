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
def _get_next_maint(self):
    """
        Returns the next Maintenance Window.

        :return:  A dictionary containing maintenance window info (or None if
                  no maintenance windows are scheduled)
                  The dictionary contains 4 keys with values of type ``str``
                      - name: The name of the maintenance window
                      - description: Description of the maintenance window
                      - beginTime: RFC3339 Timestamp
                      - endTime: RFC3339 Timestamp
        :rtype:   ``dict`` or ``None``
        """
    begin = None
    next_window = None
    if not self.maintenance_windows:
        return None
    if len(self.maintenance_windows) == 1:
        return self.maintenance_windows[0]
    for mw in self.maintenance_windows:
        begin_next = timestamp_to_datetime(mw['beginTime'])
        if not begin or begin_next < begin:
            begin = begin_next
            next_window = mw
    return next_window