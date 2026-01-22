import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_firewall_rule(self, element, locations, network_domain):
    location_id = element.get('datacenterId')
    location = list(filter(lambda x: x.id == location_id, locations))[0]
    return NttCisFirewallRule(id=element.get('id'), network_domain=network_domain, name=findtext(element, 'name', TYPES_URN), action=findtext(element, 'action', TYPES_URN), ip_version=findtext(element, 'ipVersion', TYPES_URN), protocol=findtext(element, 'protocol', TYPES_URN), enabled=findtext(element, 'enabled', TYPES_URN), source=self._to_firewall_address(element.find(fixxpath('source', TYPES_URN))), destination=self._to_firewall_address(element.find(fixxpath('destination', TYPES_URN))), location=location, status=findtext(element, 'state', TYPES_URN))