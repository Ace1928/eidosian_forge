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
def _handle_profile_context(self, tmp):
    if 'context' not in tmp:
        tmp['context'] = 'all'
    elif 'name' not in tmp:
        raise F5ModuleError('A profile name must be specified when a context is specified.')
    tmp['context'] = tmp['context'].replace('server-side', 'serverside')
    tmp['context'] = tmp['context'].replace('client-side', 'clientside')