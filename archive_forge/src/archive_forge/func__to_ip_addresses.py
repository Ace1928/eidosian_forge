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
def _to_ip_addresses(self, el):
    public_ips = [ip.get('addr') for ip in findall(findall(el, 'public', self.XML_NAMESPACE)[0], 'ip', self.XML_NAMESPACE)]
    private_ips = [ip.get('addr') for ip in findall(findall(el, 'private', self.XML_NAMESPACE)[0], 'ip', self.XML_NAMESPACE)]
    return OpenStack_1_0_NodeIpAddresses(public_ips, private_ips)