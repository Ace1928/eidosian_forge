from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def add_proxy(self, data):
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        parameters = {}
        for item in data:
            if data[item]:
                parameters[item] = data[item]
        if 'proxy_address' in data and data['status'] != '5':
            parameters.pop('proxy_address', False)
        if 'interface' in data and data['status'] != '6':
            parameters.pop('interface', False)
        proxy_ids_list = self._zapi.proxy.create(parameters)
        self._module.exit_json(changed=True, result='Successfully added proxy %s (%s)' % (data['host'], data['status']))
        if len(proxy_ids_list) >= 1:
            return proxy_ids_list['proxyids'][0]
    except Exception as e:
        self._module.fail_json(msg='Failed to create proxy %s: %s' % (data['host'], e))