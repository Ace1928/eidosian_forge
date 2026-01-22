from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def _wait_for_state(self, states):
    start = datetime.now()
    timeout = self._module.params['api_timeout'] * 2
    while datetime.now() - start < timedelta(seconds=timeout):
        server_info = self._get_server_info(refresh=True)
        if server_info.get('state') in states:
            return server_info
        sleep(1)
    if server_info.get('name') is not None:
        msg = 'Timeout while waiting for a state change on server %s to states %s. Current state is %s.' % (server_info.get('name'), states, server_info.get('state'))
    else:
        name_uuid = self._module.params.get('name') or self._module.params.get('uuid')
        msg = 'Timeout while waiting to find the server %s' % name_uuid
    self._module.fail_json(msg=msg)