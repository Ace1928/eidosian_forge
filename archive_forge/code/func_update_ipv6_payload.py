from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def update_ipv6_payload(src_dict, new_dict):
    diff = 0
    if new_dict:
        if new_dict.get('Enable') != src_dict.get('Enable'):
            src_dict['Enable'] = new_dict.get('Enable')
            diff += 1
        if new_dict.get('Enable'):
            tmp_dict = {'EnableAutoConfiguration': ['StaticIPAddress', 'StaticPrefixLength', 'StaticGateway'], 'UseDHCPForDNSServerNames': ['StaticPreferredDNSServer', 'StaticAlternateDNSServer']}
            for key, val in tmp_dict.items():
                if new_dict.get(key) is not None:
                    if new_dict.get(key) != src_dict.get(key):
                        src_dict[key] = new_dict.get(key)
                        diff += 1
                    if not new_dict.get(key):
                        diff = diff + _compare_dict_merge(src_dict, new_dict, val)
    return diff