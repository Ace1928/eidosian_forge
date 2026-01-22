from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_value(self, conditiontype, value):
    """Construct operator

        Args:
            conditiontype: type of condition to construct
            value: value to construct

        Returns:
            str: constructed value
        """
    try:
        if conditiontype == 0:
            return self._zapi_wrapper.get_hostgroup_by_hostgroup_name(value)['groupid']
        if conditiontype == 1:
            return self._zapi_wrapper.get_host_by_host_name(value)['hostid']
        if conditiontype == 2:
            return self._zapi_wrapper.get_trigger_by_trigger_name(value)['triggerid']
        if conditiontype == 4:
            return zabbix_utils.helper_to_numeric_value(['not classified', 'information', 'warning', 'average', 'high', 'disaster'], value or 'not classified')
        if conditiontype == 5:
            return zabbix_utils.helper_to_numeric_value(['ok', 'problem'], value or 'ok')
        if conditiontype == 8:
            return zabbix_utils.helper_to_numeric_value(['SSH', 'LDAP', 'SMTP', 'FTP', 'HTTP', 'POP', 'NNTP', 'IMAP', 'TCP', 'Zabbix agent', 'SNMPv1 agent', 'SNMPv2 agent', 'ICMP ping', 'SNMPv3 agent', 'HTTPS', 'Telnet'], value)
        if conditiontype == 10:
            return zabbix_utils.helper_to_numeric_value(['up', 'down', 'discovered', 'lost'], value)
        if conditiontype == 13:
            return self._zapi_wrapper.get_template_by_template_name(value)['templateid']
        if conditiontype == 16:
            return zabbix_utils.helper_to_numeric_value(['Yes', 'No'], value)
        if conditiontype == 18:
            return self._zapi_wrapper.get_discovery_rule_by_discovery_rule_name(value)['druleid']
        if conditiontype == 19:
            return self._zapi_wrapper.get_discovery_check_by_discovery_check_name(value)['dcheckid']
        if conditiontype == 20:
            return self._zapi_wrapper.get_proxy_by_proxy_name(value)['proxyid']
        if conditiontype == 21:
            return zabbix_utils.helper_to_numeric_value(['pchldrfor0', 'host', 'service'], value)
        if conditiontype == 23:
            return zabbix_utils.helper_to_numeric_value(['item in not supported state', 'item in normal state', 'LLD rule in not supported state', 'LLD rule in normal state', 'trigger in unknown state', 'trigger in normal state'], value)
        return value
    except Exception:
        self._module.fail_json(msg="Unsupported value '%s' for specified condition type.\n                       Check out Zabbix API documentation for supported values for\n                       condition type '%s' at\n                       https://www.zabbix.com/documentation/3.4/manual/api/reference/action/object#action_filter_condition" % (value, conditiontype))