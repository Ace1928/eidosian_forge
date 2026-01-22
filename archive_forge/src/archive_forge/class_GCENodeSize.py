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
class GCENodeSize(NodeSize):
    """A GCE Node Size (MachineType) class."""

    def __init__(self, id, name, ram, disk, bandwidth, price, driver, extra=None):
        self.extra = extra
        super().__init__(id, name, ram, disk, bandwidth, price, driver, extra=extra)