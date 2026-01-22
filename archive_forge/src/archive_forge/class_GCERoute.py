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
class GCERoute(UuidMixin):
    """A GCE Route object class."""

    def __init__(self, id, name, dest_range, priority, network='default', tags=None, driver=None, extra=None):
        self.id = str(id)
        self.name = name
        self.dest_range = dest_range
        self.priority = priority
        self.network = network
        self.tags = tags
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this route

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_destroy_route(route=self)

    def __repr__(self):
        network_name = getattr(self.network, 'name', self.network)
        return '<GCERoute id="{}" name="{}" dest_range="{}" network="{}">'.format(self.id, self.name, self.dest_range, network_name)