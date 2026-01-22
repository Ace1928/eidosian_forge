import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStackNodeSize(NodeSize):
    """
    NodeSize class for the OpenStack.org driver.

    Following the example of OpenNebula.org driver
    and following guidelines:
    https://issues.apache.org/jira/browse/LIBCLOUD-119
    """

    def __init__(self, id, name, ram, disk, bandwidth, price, driver, vcpus=None, ephemeral_disk=None, swap=None, extra=None):
        super().__init__(id=id, name=name, ram=ram, disk=disk, bandwidth=bandwidth, price=price, driver=driver)
        self.vcpus = vcpus
        self.ephemeral_disk = ephemeral_disk
        self.swap = swap
        self.extra = extra

    def __repr__(self):
        return '<OpenStackNodeSize: id=%s, name=%s, ram=%s, disk=%s, bandwidth=%s, price=%s, driver=%s, vcpus=%s,  ...>' % (self.id, self.name, self.ram, self.disk, self.bandwidth, self.price, self.driver.name, self.vcpus)