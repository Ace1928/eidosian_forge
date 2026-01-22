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
class OneConnectProfilesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'clientTimeout': 'client_timeout', 'defaultsFrom': 'parent', 'idleTimeoutOverride': 'idle_timeout_override', 'limitType': 'limit_type', 'maxAge': 'max_age', 'maxReuse': 'max_reuse', 'maxSize': 'max_size', 'sharePools': 'share_pools', 'sourceMask': 'source_mask'}
    returnables = ['full_path', 'name', 'parent', 'description', 'idle_timeout_override', 'limit_type', 'max_age', 'max_reuse', 'max_size', 'share_pools', 'source_mask']

    @property
    def description(self):
        if self._values['description'] in [None, 'none']:
            return None
        return self._values['description']

    @property
    def idle_timeout_override(self):
        if self._values['idle_timeout_override'] is None:
            return None
        elif self._values['idle_timeout_override'] == 'disabled':
            return 0
        elif self._values['idle_timeout_override'] == 'indefinite':
            return 4294967295
        return int(self._values['idle_timeout_override'])

    @property
    def share_pools(self):
        return flatten_boolean(self._values['share_pools'])