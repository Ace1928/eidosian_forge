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
def _verify_minimum_profile(self):
    if self.want.profiles:
        return None
    if self.want.type == 'internal' and self.want.profiles == '':
        raise F5ModuleError("An 'internal' server must have at least one profile relevant to its 'ip_protocol'. For example, 'tcp', 'udp', or variations of those.")