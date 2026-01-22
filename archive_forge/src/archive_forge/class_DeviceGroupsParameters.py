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
class DeviceGroupsParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'autoSync': 'autosync_enabled', 'asmSync': 'asm_sync_enabled', 'devicesReference': 'devices', 'fullLoadOnSync': 'full_load_on_sync', 'incrementalConfigSyncSizeMax': 'incremental_config_sync_size_maximum', 'networkFailover': 'network_failover_enabled'}
    returnables = ['full_path', 'name', 'autosync_enabled', 'description', 'devices', 'full_load_on_sync', 'incremental_config_sync_size_maximum', 'network_failover_enabled', 'type', 'asm_sync_enabled']

    @property
    def network_failover_enabled(self):
        if self._values['network_failover_enabled'] is None:
            return None
        if self._values['network_failover_enabled'] == 'enabled':
            return 'yes'
        return 'no'

    @property
    def asm_sync_enabled(self):
        if self._values['asm_sync_enabled'] is None:
            return None
        if self._values['asm_sync_enabled'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def autosync_enabled(self):
        if self._values['autosync_enabled'] is None:
            return None
        if self._values['autosync_enabled'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def full_load_on_sync(self):
        if self._values['full_load_on_sync'] is None:
            return None
        if self._values['full_load_on_sync'] == 'true':
            return 'yes'
        return 'no'

    @property
    def devices(self):
        if self._values['devices'] is None or 'items' not in self._values['devices']:
            return None
        result = [x['fullPath'] for x in self._values['devices']['items']]
        result.sort()
        return result