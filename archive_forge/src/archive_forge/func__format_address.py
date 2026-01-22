from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _format_address(self, type):
    if self._values[type] is None:
        return None
    pattern = '(?P<addr>[^%/]+)(%(?P<rd>\\d+))?(/(?P<cidr>\\d+))?'
    if '%' in self._values[type]:
        matches = re.match(pattern, self._values[type])
        if not matches:
            return None
        addr = matches.group('addr')
        if addr is None:
            return -1
        cidr = matches.group('cidr')
        rd = matches.group('rd')
        if cidr is not None:
            ip = ip_interface(u'{0}/{1}'.format(addr, cidr))
        else:
            ip = ip_interface(u'{0}'.format(addr))
        if rd:
            result = '{0}%{1}/{2}'.format(str(ip.ip), rd, ip.network.prefixlen)
        else:
            result = '{0}/{1}'.format(str(ip.ip), ip.network.prefixlen)
        return result
    return str(ip_interface(u'{0}'.format(self._values[type])))