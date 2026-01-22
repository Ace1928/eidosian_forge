from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_opgroup(self, operation):
    """Construct operation group.

        Args:
            operation: operation to construct group

        Returns:
            list: constructed operation group
        """
    return [{'groupid': self._zapi_wrapper.get_hostgroup_by_hostgroup_name(_group)['groupid']} for _group in operation.get('host_groups', [])]