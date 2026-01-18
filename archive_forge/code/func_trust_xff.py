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
@property
def trust_xff(self):
    if 'trust_xff' in self._values:
        if self._values['trust_xff'] is None:
            return None
        return flatten_boolean(self._values['trust_xff'])
    if 'general' in self._values:
        if self._values['general'] is None:
            return None
        if 'trustXff' in self._values['general']:
            return flatten_boolean(self._values['general']['trustXff'])