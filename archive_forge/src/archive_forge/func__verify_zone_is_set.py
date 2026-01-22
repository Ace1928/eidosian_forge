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
def _verify_zone_is_set(self, zone=None):
    """
        Verify that the zone / location is set for a particular operation -
        either via "datacenter" driver constructor argument or via
        "location" / "zone" keyword argument passed to the specific method
        call.

        This check is mandatory for methods which rely on the location being
        set - e.g. create_node.
        """
    if self.zone:
        return True
    if not zone:
        msg = 'Zone not provided. Zone needs to be specified for thisoperation. This can be done by passing "datacenter" argument to the driver constructor or by passing location / zone argument to this method.'
        raise ValueError(msg)
    return True