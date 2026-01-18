from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_global_macro(self, global_macro_obj, macro_name, macro_value, macro_type, macro_description):
    global_macro_id = global_macro_obj['globalmacroid']
    try:
        if global_macro_obj['type'] == '0' or global_macro_obj['type'] == '2':
            if global_macro_obj['macro'] == macro_name and global_macro_obj['value'] == macro_value and (global_macro_obj['type'] == macro_type) and (global_macro_obj['description'] == macro_description):
                self._module.exit_json(changed=False, result='Global macro %s already up to date' % macro_name)
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.usermacro.updateglobal({'globalmacroid': global_macro_id, 'macro': macro_name, 'value': macro_value, 'type': macro_type, 'description': macro_description})
        self._module.exit_json(changed=True, result='Successfully updated global macro %s' % macro_name)
    except Exception as e:
        self._module.fail_json(msg='Failed to update global macro %s: %s' % (macro_name, e))