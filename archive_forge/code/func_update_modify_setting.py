from __future__ import (absolute_import, division, print_function)
import re
import json
import codecs
import binascii
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def update_modify_setting(modify_payload, existing_payload, setting_type, sub_keys):
    """update current pool sub setting setting to modify payload if not provided
     in the options to avoid the null update from ome"""
    for sub_key in sub_keys:
        if sub_key not in modify_payload[setting_type] and sub_key in existing_payload[setting_type]:
            modify_payload[setting_type][sub_key] = existing_payload[setting_type][sub_key]
        elif existing_payload[setting_type]:
            if modify_payload[setting_type].get(sub_key) and existing_payload[setting_type].get(sub_key):
                modify_setting = modify_payload[setting_type][sub_key]
                existing_setting_payload = existing_payload[setting_type][sub_key]
                diff_item = list(set(existing_setting_payload) - set(modify_setting))
                for key in diff_item:
                    modify_payload[setting_type][sub_key][key] = existing_setting_payload[key]