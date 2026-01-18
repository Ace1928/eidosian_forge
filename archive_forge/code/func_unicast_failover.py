from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def unicast_failover(self):
    if self.want.unicast_failover == [] and self.have.unicast_failover is None:
        return None
    if self.want.unicast_failover is None:
        return None
    if self.have.unicast_failover is None:
        return self.want.unicast_failover
    want = self.to_tuple(self.want.unicast_failover)
    have = self.to_tuple(self.have.unicast_failover)
    if set(want) == set(have):
        return None
    else:
        return self.want.unicast_failover