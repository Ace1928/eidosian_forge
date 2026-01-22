from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def _init_server_container(self):
    return {'uuid': self._module.params.get('uuid') or self._info.get('uuid'), 'name': self._module.params.get('name') or self._info.get('name'), 'state': 'absent'}