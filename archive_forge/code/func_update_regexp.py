from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_regexp(self, current_regexp, name, test_string, expressions):
    try:
        current_expressions = []
        for expression in current_regexp['expressions']:
            if expression['expression_type'] != '1':
                expression = zabbix_utils.helper_normalize_data(expression, del_keys=['exp_delimiter'])[0]
            current_expressions.append(expression)
        future_expressions = self._convert_expressions_to_json(expressions)
        diff_expressions = []
        zabbix_utils.helper_compare_lists(current_expressions, future_expressions, diff_expressions)
        if current_regexp['name'] == name and current_regexp['test_string'] == test_string and (len(diff_expressions) == 0):
            self._module.exit_json(changed=False)
        else:
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.regexp.update({'regexpid': current_regexp['regexpid'], 'name': name, 'test_string': test_string, 'expressions': future_expressions})
            self._module.exit_json(changed=True, msg='Successfully updated regular expression setting.')
    except Exception as e:
        self._module.fail_json(msg='Failed to update regular expression setting: %s' % e)