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
class OpenStack_2_SubNet:
    """
    A Virtual SubNet.
    """

    def __init__(self, id, name, cidr, network_id, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.cidr = cidr
        self.network_id = network_id
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<OpenStack_2_SubNet id="{}" name="{}" cidr="{}">'.format(self.id, self.name, self.cidr)