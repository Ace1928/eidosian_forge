from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class VirtualAddressesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'arp': 'arp_enabled', 'autoDelete': 'auto_delete_enabled', 'connectionLimit': 'connection_limit', 'icmpEcho': 'icmp_echo', 'mask': 'netmask', 'routeAdvertisement': 'route_advertisement', 'trafficGroup': 'traffic_group', 'inheritedTrafficGroup': 'inherited_traffic_group'}
    returnables = ['full_path', 'name', 'address', 'arp_enabled', 'auto_delete_enabled', 'connection_limit', 'description', 'enabled', 'icmp_echo', 'floating', 'netmask', 'route_advertisement', 'traffic_group', 'spanning', 'inherited_traffic_group']

    @property
    def spanning(self):
        return flatten_boolean(self._values['spanning'])

    @property
    def arp_enabled(self):
        return flatten_boolean(self._values['arp_enabled'])

    @property
    def route_advertisement(self):
        return flatten_boolean(self._values['route_advertisement'])

    @property
    def auto_delete_enabled(self):
        return flatten_boolean(self._values['auto_delete_enabled'])

    @property
    def inherited_traffic_group(self):
        return flatten_boolean(self._values['inherited_traffic_group'])

    @property
    def icmp_echo(self):
        return flatten_boolean(self._values['icmp_echo'])

    @property
    def floating(self):
        return flatten_boolean(self._values['floating'])

    @property
    def enabled(self):
        return flatten_boolean(self._values['enabled'])