from __future__ import (absolute_import, division, print_function)
import copy
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def update_prefix_length(req_payload):
    prefix_length = req_payload.get('PrefixLength')
    if prefix_length == '0':
        req_payload['PrefixLength'] = ''