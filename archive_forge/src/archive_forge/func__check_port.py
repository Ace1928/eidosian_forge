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
def _check_port(self):
    try:
        port = int(self._values['port'])
    except ValueError:
        raise F5ModuleError('The specified port was not a valid integer')
    if 0 <= port <= 65535:
        return port
    raise F5ModuleError('Valid ports must be in range 0 - 65535')