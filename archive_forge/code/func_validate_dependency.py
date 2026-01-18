from __future__ import (absolute_import, division, print_function)
import json
import socket
import copy
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def validate_dependency(mparams):
    params = copy.deepcopy(mparams)
    ipv4 = params.get('ipv4_configuration')
    if ipv4:
        rm_list = []
        dhcp = ['static_preferred_dns_server', 'static_alternate_dns_server']
        static = ['static_ip_address', 'static_gateway', 'static_subnet_mask']
        bools = ['enable_dhcp', 'use_dhcp_to_obtain_dns_server_address']
        if ipv4.get('use_dhcp_to_obtain_dns_server_address') is True:
            rm_list.extend(dhcp)
        if ipv4.get('enable_dhcp') is True:
            rm_list.extend(static)
        if ipv4.get('enable_ipv4') is False:
            rm_list.extend(dhcp)
            rm_list.extend(static)
            rm_list.extend(bools)
        for prm in rm_list:
            ipv4.pop(prm, None)
    ipv6 = params.get('ipv6_configuration')
    if ipv6:
        rm_list = []
        dhcp = ['static_preferred_dns_server', 'static_alternate_dns_server']
        static = ['static_ip_address', 'static_gateway', 'static_prefix_length']
        bools = ['enable_auto_configuration', 'use_dhcpv6_to_obtain_dns_server_address']
        if ipv6.get('use_dhcpv6_to_obtain_dns_server_address') is True:
            rm_list.extend(dhcp)
        if ipv6.get('enable_auto_configuration') is True:
            rm_list.extend(static)
        if ipv6.get('enable_ipv6') is False:
            rm_list.extend(dhcp)
            rm_list.extend(static)
            rm_list.extend(bools)
        for prm in rm_list:
            ipv6.pop(prm, None)
    vlan = params.get('management_vlan')
    if vlan:
        if vlan.get('enable_vlan') is False:
            vlan.pop('vlan_id', None)
    dns = params.get('dns_configuration')
    if dns:
        if dns.get('auto_negotiation') is True:
            dns.pop('network_speed', None)
        if dns.get('use_dhcp_for_dns_domain_name') is True:
            dns.pop('dns_domain_name', None)
    return params