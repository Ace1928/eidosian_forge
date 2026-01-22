from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class Dchecks(ZabbixBase):
    """
    Restructures the user defined discovery checks to fit the Zabbix API requirements
    """

    def construct_the_data(self, _dchecks):
        """Construct the user defined discovery check to fit the Zabbix API
        requirements
        Args:
            _dchecks: discovery checks to construct
        Returns:
            dict: user defined discovery checks
        """
        if _dchecks is None:
            return None
        constructed_data = []
        for check in _dchecks:
            constructed_check = {'type': zabbix_utils.helper_to_numeric_value(['SSH', 'LDAP', 'SMTP', 'FTP', 'HTTP', 'POP', 'NNTP', 'IMAP', 'TCP', 'Zabbix', 'SNMPv1', 'SNMPv2', 'ICMP', 'SNMPv3', 'HTTPS', 'Telnet'], check.get('type')), 'uniq': int(check.get('uniq'))}
            constructed_check.update({'host_source': zabbix_utils.helper_to_numeric_value(['None', 'DNS', 'IP', 'discovery'], check.get('host_source')), 'name_source': zabbix_utils.helper_to_numeric_value(['None', 'DNS', 'IP', 'discovery'], check.get('name_source'))})
            if constructed_check['type'] in (0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15):
                constructed_check['ports'] = check.get('ports')
            if constructed_check['type'] == 9:
                constructed_check['ports'] = check.get('ports')
                constructed_check['key_'] = check.get('key')
            if constructed_check['type'] in (10, 11):
                constructed_check['ports'] = check.get('ports')
                constructed_check['snmp_community'] = check.get('snmp_community')
                constructed_check['key_'] = check.get('key')
            if constructed_check['type'] == 13:
                constructed_check['ports'] = check.get('ports')
                constructed_check['key_'] = check.get('key')
                constructed_check['snmpv3_contextname'] = check.get('snmpv3_contextname')
                constructed_check['snmpv3_securityname'] = check.get('snmpv3_securityname')
                constructed_check['snmpv3_securitylevel'] = zabbix_utils.helper_to_numeric_value(['noAuthNoPriv', 'authNoPriv', 'authPriv'], check.get('snmpv3_securitylevel'))
                if constructed_check['snmpv3_securitylevel'] in (1, 2):
                    constructed_check['snmpv3_authprotocol'] = zabbix_utils.helper_to_numeric_value(['MD5', 'SHA'], check.get('snmpv3_authprotocol'))
                    constructed_check['snmpv3_authpassphrase'] = check.get('snmpv3_authpassphrase')
                if constructed_check['snmpv3_securitylevel'] == 2:
                    constructed_check['snmpv3_privprotocol'] = zabbix_utils.helper_to_numeric_value(['DES', 'AES'], check.get('snmpv3_privprotocol'))
                    constructed_check['snmpv3_privpassphrase'] = check.get('snmpv3_privpassphrase')
            constructed_data.append(constructed_check)
        return zabbix_utils.helper_cleanup_data(constructed_data)