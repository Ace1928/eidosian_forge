from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_script(self, script_id, name, script_type, command, scope, execute_on, menu_path, authtype, username, password, publickey, privatekey, port, host_group, user_group, host_access, confirmation, script_timeout, parameters, description):
    generated_config = self.generate_script_config(name, script_type, command, scope, execute_on, menu_path, authtype, username, password, publickey, privatekey, port, host_group, user_group, host_access, confirmation, script_timeout, parameters, description)
    live_config = self._zapi.script.get({'filter': {'name': name}})[0]
    change_parameters = {}
    difference = zabbix_utils.helper_cleanup_data(zabbix_utils.helper_compare_dictionaries(generated_config, live_config, change_parameters))
    if not difference:
        self._module.exit_json(changed=False, msg='Script %s up to date' % name)
    if self._module.check_mode:
        self._module.exit_json(changed=True)
    generated_config['scriptid'] = live_config['scriptid']
    self._zapi.script.update(generated_config)
    self._module.exit_json(changed=True, msg='Script %s updated' % name)