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
class OpenStack_2_PortInterfaceState(Type):
    """
    Standard states of OpenStack_2_PortInterfaceState
    """
    BUILD = 'build'
    ACTIVE = 'active'
    DOWN = 'down'
    UNKNOWN = 'unknown'