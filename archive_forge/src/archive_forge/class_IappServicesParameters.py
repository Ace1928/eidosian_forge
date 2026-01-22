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
class IappServicesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'deviceGroup': 'device_group', 'inheritedDevicegroup': 'inherited_device_group', 'inheritedTrafficGroup': 'inherited_traffic_group', 'strictUpdates': 'strict_updates', 'templateModified': 'template_modified', 'trafficGroup': 'traffic_group'}
    returnables = ['full_path', 'name', 'device_group', 'inherited_device_group', 'inherited_traffic_group', 'strict_updates', 'template_modified', 'traffic_group', 'tables', 'variables', 'metadata', 'lists', 'description']

    @property
    def description(self):
        if self._values['description'] in [None, 'none']:
            return None
        return self._values['description']

    @property
    def inherited_device_group(self):
        return flatten_boolean(self._values['inherited_device_group'])

    @property
    def inherited_traffic_group(self):
        return flatten_boolean(self._values['inherited_traffic_group'])

    @property
    def strict_updates(self):
        return flatten_boolean(self._values['strict_updates'])

    @property
    def template_modified(self):
        return flatten_boolean(self._values['template_modified'])