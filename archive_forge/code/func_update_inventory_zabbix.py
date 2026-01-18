from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_inventory_zabbix(self, host_id, inventory):
    if not inventory:
        return
    request_str = {'hostid': host_id, 'inventory': inventory}
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.host.update(request_str)
    except Exception as e:
        self._module.fail_json(msg='Failed to set inventory to host: %s' % e)