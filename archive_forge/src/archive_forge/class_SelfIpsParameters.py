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
class SelfIpsParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'trafficGroup': 'traffic_group', 'servicePolicy': 'service_policy', 'allowService': 'allow_access_list', 'inheritedTrafficGroup': 'traffic_group_inherited'}
    returnables = ['full_path', 'name', 'address', 'description', 'netmask', 'netmask_cidr', 'floating', 'traffic_group', 'service_policy', 'vlan', 'allow_access_list', 'traffic_group_inherited']

    @property
    def address(self):
        parts = self._values['address'].split('/')
        return parts[0]

    @property
    def netmask(self):
        result = None
        parts = self._values['address'].split('/')
        if is_valid_ip(parts[0]):
            ip = ip_interface(u'{0}'.format(self._values['address']))
            result = ip.netmask
        return str(result)

    @property
    def netmask_cidr(self):
        parts = self._values['address'].split('/')
        return int(parts[1])

    @property
    def traffic_group_inherited(self):
        if self._values['traffic_group_inherited'] is None:
            return None
        elif self._values['traffic_group_inherited'] in [False, 'false']:
            return 'no'
        else:
            return 'yes'

    @property
    def floating(self):
        if self._values['floating'] is None:
            return None
        elif self._values['floating'] == 'disabled':
            return 'no'
        else:
            return 'yes'