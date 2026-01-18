from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def validate_want(self, want, state):
    if state == 'deleted':
        if 'servers' in want and want['servers'] is not None:
            for server in want['servers']:
                key_id_config = server.get('key_id', None)
                minpoll_config = server.get('minpoll', None)
                maxpoll_config = server.get('maxpoll', None)
                prefer_config = server.get('prefer', None)
                if key_id_config or minpoll_config or maxpoll_config or (prefer_config is not None):
                    err_msg = 'NTP server parameter(s) can not be deleted.'
                    self._module.fail_json(msg=err_msg, code=405)
        if 'ntp_keys' in want and want['ntp_keys'] is not None:
            for ntp_key in want['ntp_keys']:
                encrypted_config = ntp_key.get('encrypted', None)
                key_type_config = ntp_key.get('key_type', None)
                key_value_config = ntp_key.get('key_value', None)
                if encrypted_config or key_type_config or key_value_config:
                    err_msg = 'NTP ntp_key parameter(s) can not be deleted.'
                    self._module.fail_json(msg=err_msg, code=405)