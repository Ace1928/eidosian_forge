from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def update_dns_payload(src_dict, new_dict):
    diff = 0
    if new_dict:
        mkey = 'RegisterWithDNS'
        if new_dict.get(mkey) is not None:
            if new_dict.get(mkey) != src_dict.get(mkey):
                src_dict[mkey] = new_dict.get(mkey)
                diff += 1
            if new_dict.get(mkey) is True:
                diff = diff + _compare_dict_merge(src_dict, new_dict, ['DnsName'])
        mkey = 'UseDHCPForDNSDomainName'
        if new_dict.get(mkey) is not None:
            if new_dict.get(mkey) != src_dict.get(mkey):
                src_dict[mkey] = new_dict.get(mkey)
                diff += 1
            if not new_dict.get(mkey):
                diff = diff + _compare_dict_merge(src_dict, new_dict, ['DnsDomainName'])
    return diff