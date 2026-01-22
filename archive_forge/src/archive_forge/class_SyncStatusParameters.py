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
class SyncStatusParameters(BaseParameters):
    api_map = {}
    returnables = ['color', 'details', 'mode', 'recommended_action', 'status', 'summary']

    @property
    def color(self):
        result = self._values.get('color', {}).get('description', '')
        if result.strip():
            return result
        return ''

    @property
    def details(self):
        result = []
        details = self._values.get('https://localhost/mgmt/tm/cm/syncStatus/0/details', {}).get('nestedStats', {}).get('entries', {})
        for entry in details.keys():
            result.append(details[entry].get('nestedStats', {}).get('entries', {}).get('details', {}).get('description', ''))
        result.reverse()
        return result

    @property
    def mode(self):
        result = self._values.get('mode', {}).get('description', '')
        if result.strip():
            return result
        return ''

    @property
    def status(self):
        result = self._values.get('status', {}).get('description', '')
        if result.strip():
            return result
        return ''

    @property
    def summary(self):
        result = self._values.get('summary', {}).get('description', '')
        if result.strip():
            return result
        return ''

    @property
    def recommended_action(self):
        for entry in self.details:
            match = re.match('.*[Rr]ecommended action:\\s(.*)$', entry)
            if match:
                return match.group(1)
        return ''