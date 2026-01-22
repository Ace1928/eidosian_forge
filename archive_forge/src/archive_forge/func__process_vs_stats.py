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
def _process_vs_stats(self, link):
    result = dict()
    item = self._read_virtual_stats_from_device(urlparse(link).path)
    if not item:
        return result
    result['status'] = item['status']['availabilityState']
    result['status_reason'] = item['status']['statusReason']
    result['state'] = item['status']['enabledState']
    result['bits_per_sec_in'] = item['metrics']['bitsPerSecIn']
    result['bits_per_sec_in'] = item['metrics']['bitsPerSecOut']
    result['pkts_per_sec_in'] = item['metrics']['pktsPerSecIn']
    result['pkts_per_sec_out'] = item['metrics']['pktsPerSecOut']
    result['connections'] = item['metrics']['connections']
    result['picks'] = item['picks']
    result['virtual_server_score'] = item['metrics']['vsScore']
    result['uptime'] = item['uptime']
    return result