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
def update_chassis_payload(module, payload):
    ipv4 = {'enable_dhcp': 'EnableDHCP', 'enable_ipv4': 'EnableIPv4', 'static_alternate_dns_server': 'StaticAlternateDNSServer', 'static_gateway': 'StaticGateway', 'static_ip_address': 'StaticIPAddress', 'static_preferred_dns_server': 'StaticPreferredDNSServer', 'static_subnet_mask': 'StaticSubnetMask', 'use_dhcp_to_obtain_dns_server_address': 'UseDHCPObtainDNSServerAddresses'}
    ipv6 = {'enable_auto_configuration': 'EnableAutoconfiguration', 'enable_ipv6': 'EnableIPv6', 'static_alternate_dns_server': 'StaticAlternateDNSServer', 'static_gateway': 'StaticGateway', 'static_ip_address': 'StaticIPv6Address', 'static_preferred_dns_server': 'StaticPreferredDNSServer', 'static_prefix_length': 'StaticPrefixLength', 'use_dhcpv6_to_obtain_dns_server_address': 'UseDHCPv6ObtainDNSServerAddresses'}
    dns = {'auto_negotiation': 'AutoNegotiation', 'dns_domain_name': 'DnsDomainName', 'dns_name': 'DnsName', 'network_speed': 'NetworkSpeed', 'register_with_dns': 'RegisterDNS', 'use_dhcp_for_dns_domain_name': 'UseDHCPForDomainName'}
    vlan = {'enable_vlan': 'EnableVLAN', 'vlan_id': 'MgmtVLANId'}
    gnrl = payload.get('GeneralSettings')
    diff = {}
    mparams = validate_dependency(module.params)
    enable_nic = mparams.get('enable_nic')
    delay = mparams.get('delay')
    if enable_nic:
        if mparams.get('ipv4_configuration'):
            df = transform_diff(mparams.get('ipv4_configuration'), ipv4, payload.get('Ipv4Settings'))
            diff.update(df)
        if mparams.get('ipv6_configuration'):
            df = transform_diff(mparams.get('ipv6_configuration'), ipv6, payload.get('Ipv6Settings'))
            diff.update(df)
        if mparams.get('dns_configuration'):
            df = transform_diff(mparams.get('dns_configuration'), dns, payload.get('GeneralSettings'))
            diff.update(df)
        if mparams.get('management_vlan'):
            df = transform_diff(mparams.get('management_vlan'), vlan, payload)
            diff.update(df)
    if gnrl.get('EnableNIC') != enable_nic:
        gnrl['EnableNIC'] = enable_nic
        diff.update({'EnableNIC': enable_nic})
    if delay != gnrl.get('Delay'):
        gnrl['Delay'] = delay
        diff.update({'Delay': delay})
    return diff