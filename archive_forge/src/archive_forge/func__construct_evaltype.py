from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_evaltype(self, _eval_type, _formula, _conditions):
    """Construct the eval type

        Args:
            _formula: zabbix condition evaluation formula
            _conditions: list of conditions to check

        Returns:
            dict: constructed acknowledge operations data
        """
    if len(_conditions) <= 1:
        return {'evaltype': '0', 'formula': None}
    if _eval_type == 'andor':
        return {'evaltype': '0', 'formula': None}
    if _eval_type == 'and':
        return {'evaltype': '1', 'formula': None}
    if _eval_type == 'or':
        return {'evaltype': '2', 'formula': None}
    if _eval_type == 'custom_expression':
        if _formula is not None:
            return {'evaltype': '3', 'formula': _formula}
        else:
            self._module.fail_json(msg="'formula' is required when 'eval_type' is set to 'custom_expression'")
    if _formula is not None:
        return {'evaltype': '3', 'formula': _formula}
    return {'evaltype': '0', 'formula': None}