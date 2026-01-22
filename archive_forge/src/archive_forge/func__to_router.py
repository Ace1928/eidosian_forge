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
def _to_router(self, obj):
    extra = {}
    extra['external_gateway_info'] = obj['external_gateway_info']
    extra['routes'] = obj['routes']
    return OpenStack_2_Router(id=obj['id'], name=obj['name'], status=obj['status'], driver=self, extra=extra)