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
class OpenStack_1_0_Connection(OpenStackComputeConnection):
    responseCls = OpenStack_1_0_Response
    default_content_type = 'application/xml; charset=UTF-8'
    accept_format = 'application/xml'
    XML_NAMESPACE = 'http://docs.rackspacecloud.com/servers/api/v1.0'