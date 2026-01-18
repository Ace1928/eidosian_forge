from __future__ import (absolute_import, division, print_function)
import copy
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def update_ipv4_data(req_data, ipv4_enabled, ipv4_enabled_deploy, ipv4_nt_deploy, deploy_options):
    if ipv4_enabled is not None and ipv4_enabled is True or (ipv4_enabled_deploy is not None and ipv4_enabled_deploy is True):
        req_data['ProtocolTypeV4'] = None
        if ipv4_enabled is not None:
            req_data['ProtocolTypeV4'] = str(ipv4_enabled).lower()
        ipv4_network_type = deploy_options.get('ipv4_network_type')
        req_data['NetworkTypeV4'] = ipv4_network_type
        if ipv4_network_type == 'Static' or (ipv4_nt_deploy is not None and ipv4_nt_deploy == 'Static'):
            req_data['IpV4SubnetMask'] = deploy_options.get('ipv4_subnet_mask')
            req_data['IpV4Gateway'] = deploy_options.get('ipv4_gateway')
    elif ipv4_enabled is not None and ipv4_enabled is False:
        req_data['ProtocolTypeV4'] = str(ipv4_enabled).lower()