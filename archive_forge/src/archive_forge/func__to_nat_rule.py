import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_nat_rule(self, element, network_domain):
    return NttCisNatRule(id=element.get('id'), network_domain=network_domain, internal_ip=findtext(element, 'internalIp', TYPES_URN), external_ip=findtext(element, 'externalIp', TYPES_URN), status=findtext(element, 'state', TYPES_URN))