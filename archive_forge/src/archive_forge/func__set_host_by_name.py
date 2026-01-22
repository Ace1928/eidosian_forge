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
def _set_host_by_name(self):
    if is_valid_ip(self.want.name):
        self.want.update({'fqdn': None, 'address': self.want.name})
    else:
        if not is_valid_hostname(self.want.name):
            raise F5ModuleError("'name' is neither a valid IP address or FQDN name.")
        self.want.update({'fqdn': self.want.name, 'address': None})