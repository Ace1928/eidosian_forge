from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_exist_interfaces_with_defaults(exist_interfaces):
    new_exist_interfaces = []
    default_interface = {'main': '0', 'useip': '0', 'ip': '', 'dns': '', 'port': ''}
    default_interface_details = {'version': 2, 'bulk': 1, 'community': '', 'securityname': '', 'contextname': '', 'securitylevel': 0, 'authprotocol': 0, 'authpassphrase': '', 'privprotocol': 0, 'privpassphrase': ''}
    for interface in exist_interfaces:
        new_interface = default_interface.copy()
        new_interface.update(interface)
        new_interface['details'] = default_interface_details.copy()
        if 'details' in interface:
            new_interface['details'].update(interface['details'])
        new_exist_interfaces.append(new_interface)
    return new_exist_interfaces