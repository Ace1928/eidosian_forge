from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_rd(self, address):
    pattern = '(?P<ip>[^%]+)%(?P<route_domain>[0-9]+)'
    matches = re.search(pattern, address)
    if matches:
        addr = matches.group('ip')
        rd = matches.group('route_domain')
        return (addr, rd)
    return (None, None)