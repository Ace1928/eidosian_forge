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
class UsersParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'partitionAccess': 'partition_access'}
    returnables = ['full_path', 'name', 'description', 'partition_access', 'shell']

    @property
    def partition_access(self):
        result = []
        if self._values['partition_access'] is None:
            return []
        for partition in self._values['partition_access']:
            del partition['nameReference']
            result.append(partition)
        return result

    @property
    def shell(self):
        if self._values['shell'] in [None, 'none']:
            return None
        return self._values['shell']