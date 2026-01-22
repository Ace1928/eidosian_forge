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
class RemoteSyslogParameters(BaseParameters):
    api_map = {'remoteServers': 'servers'}
    returnables = ['servers']

    def _morph_keys(self, key_map, item):
        for k, v in iteritems(key_map):
            item[v] = item.pop(k, None)
        result = self._filter_params(item)
        return result

    def _format_servers(self, items):
        result = list()
        key_map = {'name': 'name', 'remotePort': 'remote_port', 'localIp': 'local_ip', 'host': 'remote_host'}
        for item in items:
            output = self._morph_keys(key_map, item)
            result.append(output)
        return result

    @property
    def servers(self):
        if self._values['servers'] is None:
            return None
        return self._format_servers(self._values['servers'])