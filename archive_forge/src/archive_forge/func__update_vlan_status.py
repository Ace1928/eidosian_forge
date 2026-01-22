from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _update_vlan_status(self, result):
    if self.want.vlans_disabled is not None:
        if self.want.vlans_disabled != self.have.vlans_disabled:
            result['vlans_disabled'] = self.want.vlans_disabled
            result['vlans_enabled'] = not self.want.vlans_disabled
    elif self.want.vlans_enabled is not None:
        if any((x.lower().endswith('/all') for x in self.want.vlans)):
            if self.have.vlans_enabled is True:
                result['vlans_disabled'] = True
                result['vlans_enabled'] = False
        elif self.want.vlans_enabled != self.have.vlans_enabled:
            result['vlans_disabled'] = not self.want.vlans_enabled
            result['vlans_enabled'] = self.want.vlans_enabled