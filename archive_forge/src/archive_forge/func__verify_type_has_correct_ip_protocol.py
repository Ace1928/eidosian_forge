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
def _verify_type_has_correct_ip_protocol(self):
    if self.want.ip_protocol is None:
        return
    if self.want.type == 'standard':
        if self.want.ip_protocol not in [6, 17, 132, 51, 50, 'any']:
            raise F5ModuleError("The 'standard' server type does not support the specified 'ip_protocol'.")
    elif self.want.type == 'performance-http':
        if self.want.ip_protocol not in [6]:
            raise F5ModuleError("The 'performance-http' server type does not support the specified 'ip_protocol'.")
    elif self.want.type == 'stateless':
        if self.want.ip_protocol not in [17]:
            raise F5ModuleError("The 'stateless' server type does not support the specified 'ip_protocol'.")
    elif self.want.type == 'dhcp':
        if self.want.ip_protocol is not None:
            raise F5ModuleError("The 'dhcp' server type does not support an 'ip_protocol'.")
    elif self.want.type == 'internal':
        if self.want.ip_protocol not in [6, 17]:
            raise F5ModuleError("The 'internal' server type does not support the specified 'ip_protocol'.")
    elif self.want.type == 'message-routing':
        if self.want.ip_protocol not in [6, 17, 132, 'all', 'any']:
            raise F5ModuleError("The 'message-routing' server type does not support the specified 'ip_protocol'.")