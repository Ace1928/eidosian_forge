from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _is_time(self, time):
    pattern = re.compile('^(\\d+)([smhdwMy]?)$')
    search_result = pattern.search(time)
    if search_result is None:
        self._module.fail_json(msg='{0} is invalid value.'.format(time))
    return True