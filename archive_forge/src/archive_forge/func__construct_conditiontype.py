from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_conditiontype(self, _condition):
    """Construct the condition type

        Args:
            _condition: condition to check

        Returns:
            str: constructed condition type data
        """
    try:
        return zabbix_utils.helper_to_numeric_value(['host_group', 'host', 'trigger', 'trigger_name', 'trigger_severity', 'trigger_value', 'time_period', 'host_ip', 'discovered_service_type', 'discovered_service_port', 'discovery_status', 'uptime_or_downtime_duration', 'received_value', 'host_template', None, None, 'maintenance_status', None, 'discovery_rule', 'discovery_check', 'proxy', 'discovery_object', 'host_name', 'event_type', 'host_metadata', 'event_tag', 'event_tag_value'], _condition['type'])
    except Exception:
        self._module.fail_json(msg="Unsupported value '%s' for condition type." % _condition['type'])