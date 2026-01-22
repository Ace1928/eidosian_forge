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
class NodesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'connectionLimit': 'connection_limit', 'dynamicRatio': 'dynamic_ratio', 'rateLimit': 'rate_limit', 'monitor': 'monitors'}
    returnables = ['full_path', 'name', 'ratio', 'description', 'connection_limit', 'address', 'dynamic_ratio', 'rate_limit', 'monitor_status', 'session_status', 'availability_status', 'enabled_status', 'status_reason', 'monitor_rule', 'monitors', 'monitor_type', 'fqdn_name', 'fqdn_auto_populate', 'fqdn_address_type', 'fqdn_up_interval', 'fqdn_down_interval']

    @property
    def fqdn_name(self):
        if self._values['fqdn'] is None:
            return None
        return self._values['fqdn'].get('tmName', None)

    @property
    def fqdn_auto_populate(self):
        if self._values['fqdn'] is None:
            return None
        return flatten_boolean(self._values['fqdn'].get('autopopulate', None))

    @property
    def fqdn_address_type(self):
        if self._values['fqdn'] is None:
            return None
        return self._values['fqdn'].get('addressFamily', None)

    @property
    def fqdn_up_interval(self):
        if self._values['fqdn'] is None:
            return None
        result = self._values['fqdn'].get('interval', None)
        if result:
            try:
                return int(result)
            except ValueError:
                return result

    @property
    def fqdn_down_interval(self):
        if self._values['fqdn'] is None:
            return None
        result = self._values['fqdn'].get('downInterval', None)
        if result:
            try:
                return int(result)
            except ValueError:
                return result

    @property
    def monitors(self):
        if self._values['monitors'] is None:
            return []
        try:
            result = re.findall('/\\w+/[^\\s}]+', self._values['monitors'])
            return result
        except Exception:
            return [self._values['monitors']]

    @property
    def monitor_type(self):
        if self._values['monitors'] is None:
            return None
        pattern = 'min\\s+\\d+\\s+of'
        matches = re.search(pattern, self._values['monitors'])
        if matches:
            return 'm_of_n'
        else:
            return 'and_list'

    @property
    def rate_limit(self):
        if self._values['rate_limit'] is None:
            return None
        elif self._values['rate_limit'] == 'disabled':
            return 0
        else:
            return int(self._values['rate_limit'])

    @property
    def monitor_status(self):
        return self._values['stats']['monitorStatus']

    @property
    def session_status(self):
        return self._values['stats']['sessionStatus']

    @property
    def availability_status(self):
        return self._values['stats']['status']['availabilityState']

    @property
    def enabled_status(self):
        return self._values['stats']['status']['enabledState']

    @property
    def status_reason(self):
        return self._values['stats']['status']['statusReason']

    @property
    def monitor_rule(self):
        return self._values['stats']['monitorRule']