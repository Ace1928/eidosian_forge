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
def validate_modify_create_payload(setting_payload, module, action):
    for key, val in setting_payload.items():
        if key in ['EthernetSettings', 'FcoeSettings'] and val:
            sub_config = val.get('Mac')
            if sub_config is None or not all([sub_config.get('IdentityCount'), sub_config.get('StartingMacAddress')]):
                module.fail_json(msg='Both starting MAC address and identity count is required to {0} an identity pool using {1} settings.'.format(action, ''.join(key.split('Settings'))))
        elif key == 'FcSettings' and val:
            sub_config = val.get('Wwnn')
            if sub_config is None or not all([sub_config.get('IdentityCount'), sub_config.get('StartingAddress')]):
                module.fail_json(msg='Both starting MAC address and identity count is required to {0} an identity pool using Fc settings.'.format(action))
        elif key == 'IscsiSettings' and val:
            sub_config1 = val.get('Mac')
            sub_config2 = val.get('InitiatorIpPoolSettings')
            if sub_config1 is None or not all([sub_config1.get('IdentityCount'), sub_config1.get('StartingMacAddress')]):
                module.fail_json(msg='Both starting MAC address and identity count is required to {0} an identity pool using {1} settings.'.format(action, ''.join(key.split('Settings'))))
            elif sub_config2:
                if not all([sub_config2.get('IpRange'), sub_config2.get('SubnetMask')]):
                    module.fail_json(msg='Both ip range and subnet mask in required to {0} an identity pool using iSCSI settings.'.format(action))