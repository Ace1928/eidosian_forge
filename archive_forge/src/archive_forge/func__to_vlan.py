import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_vlan(self, element, locations):
    location_id = element.get('datacenterId')
    location = list(filter(lambda x: x.id == location_id, locations))[0]
    ip_range = element.find(fixxpath('privateIpv4Range', TYPES_URN))
    ip6_range = element.find(fixxpath('ipv6Range', TYPES_URN))
    network_domain_el = element.find(fixxpath('networkDomain', TYPES_URN))
    network_domain = self.ex_get_network_domain(network_domain_el.get('id'))
    return NttCisVlan(id=element.get('id'), name=findtext(element, 'name', TYPES_URN), description=findtext(element, 'description', TYPES_URN), network_domain=network_domain, private_ipv4_range_address=ip_range.get('address'), private_ipv4_range_size=int(ip_range.get('prefixSize')), ipv6_range_address=ip6_range.get('address'), ipv6_range_size=int(ip6_range.get('prefixSize')), ipv4_gateway=findtext(element, 'ipv4GatewayAddress', TYPES_URN), ipv6_gateway=findtext(element, 'ipv6GatewayAddress', TYPES_URN), location=location, status=findtext(element, 'state', TYPES_URN))