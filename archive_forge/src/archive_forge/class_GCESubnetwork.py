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
class GCESubnetwork(UuidMixin):
    """A GCE Subnetwork object class."""

    def __init__(self, id, name, cidr, network, region, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.cidr = cidr
        self.network = network
        self.region = region
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this subnetwork

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_destroy_subnetwork(self)

    def __repr__(self):
        return '<GCESubnetwork id="%s" name="%s" region="%s" network="%s" cidr="%s">' % (self.id, self.name, self.region.name, self.network.name, self.cidr)