from __future__ import absolute_import, division, print_function
import os
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip, validate_ip_v6_address
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _update_api_state_attributes(self):
    if self.want.state == 'forced_offline':
        self.want.update({'state': 'user-down', 'session': 'user-disabled'})
    elif self.want.state == 'disabled':
        self.want.update({'state': 'user-up', 'session': 'user-disabled'})
    elif self.want.state in ['present', 'enabled']:
        self.want.update({'state': 'user-up', 'session': 'user-enabled'})