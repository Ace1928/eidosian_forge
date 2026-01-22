from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_operationtype(self, operation):
    """Construct operation type.

        Args:
            operation: operation to construct type

        Returns:
            str: constructed operation type
        """
    try:
        return zabbix_utils.helper_to_numeric_value(['send_message', 'remote_command', None, None, None, None, None, None, None, None, None, None, 'notify_all_involved'], operation['type'])
    except Exception:
        self._module.fail_json(msg="Unsupported value '%s' for acknowledge operation type." % operation['type'])