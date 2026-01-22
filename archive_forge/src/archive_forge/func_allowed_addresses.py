from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ipaddress import ip_network
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def allowed_addresses(self):
    if self.want.allowed_addresses is None:
        return None
    if self.have.allowed_addresses is None:
        if self.want.allowed_addresses:
            return self.want.allowed_addresses
        return None
    want = set(self.want.allowed_addresses)
    have = set(self.have.allowed_addresses)
    if want != have:
        result = list(want)
        result.sort()
        return result