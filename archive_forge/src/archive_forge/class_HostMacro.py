from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class HostMacro(ZabbixBase):

    def get_host_id(self, host_name):
        try:
            host_list = self._zapi.host.get({'output': 'extend', 'filter': {'host': host_name}})
            if len(host_list) < 1:
                self._module.fail_json(msg='Host not found: %s' % host_name)
            else:
                host_id = host_list[0]['hostid']
                return host_id
        except Exception as e:
            self._module.fail_json(msg='Failed to get the host %s id: %s.' % (host_name, e))

    def get_host_macro(self, macro_name, host_id):
        try:
            host_macro_list = self._zapi.usermacro.get({'output': 'extend', 'selectSteps': 'extend', 'hostids': [host_id], 'filter': {'macro': macro_name}})
            if len(host_macro_list) > 0:
                return host_macro_list[0]
            return None
        except Exception as e:
            self._module.fail_json(msg='Failed to get host macro %s: %s' % (macro_name, e))

    def create_host_macro(self, macro_name, macro_value, macro_type, macro_description, host_id):
        try:
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.usermacro.create({'hostid': host_id, 'macro': macro_name, 'value': macro_value, 'type': macro_type, 'description': macro_description})
            self._module.exit_json(changed=True, result='Successfully added host macro %s' % macro_name)
        except Exception as e:
            self._module.fail_json(msg='Failed to create host macro %s: %s' % (macro_name, e))

    def update_host_macro(self, host_macro_obj, macro_name, macro_value, macro_type, macro_description):
        host_macro_id = host_macro_obj['hostmacroid']
        if host_macro_obj['macro'] == macro_name:
            if host_macro_obj['type'] == '0' and macro_type == '0' and (host_macro_obj['value'] == macro_value) and (host_macro_obj['description'] == macro_description):
                self._module.exit_json(changed=False, result='Host macro %s already up to date' % macro_name)
        try:
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.usermacro.update({'hostmacroid': host_macro_id, 'value': macro_value, 'type': macro_type, 'description': macro_description})
            self._module.exit_json(changed=True, result='Successfully updated host macro %s' % macro_name)
        except Exception as e:
            self._module.fail_json(msg='Failed to update host macro %s: %s' % (macro_name, e))

    def delete_host_macro(self, host_macro_obj, macro_name):
        host_macro_id = host_macro_obj['hostmacroid']
        try:
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.usermacro.delete([host_macro_id])
            self._module.exit_json(changed=True, result='Successfully deleted host macro %s' % macro_name)
        except Exception as e:
            self._module.fail_json(msg='Failed to delete host macro %s: %s' % (macro_name, e))