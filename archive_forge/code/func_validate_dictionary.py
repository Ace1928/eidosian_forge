from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def validate_dictionary(module, loc_resp):
    data_center = module.params.get('data_center')
    room = module.params.get('room')
    aisle = module.params.get('aisle')
    rack = module.params.get('rack')
    rack_slot = module.params.get('rack_slot')
    location = module.params.get('location')
    req_dict = {'DataCenter': data_center, 'Room': room, 'Aisle': aisle, 'RackName': rack, 'Location': location}
    req_filter_none = dict(((k, v) for k, v in req_dict.items() if v is not None))
    keys = list(req_filter_none.keys())
    exit_dict = dict(((k, v) for k, v in loc_resp.items() if k in keys and v is not None))
    if rack_slot is not None:
        req_dict.update({'RackSlot': rack_slot})
        req_filter_none.update({'RackSlot': rack_slot})
        exit_dict.update({'RackSlot': loc_resp['RackSlot']})
    diff = bool(set(req_filter_none.items()) ^ set(exit_dict.items()))
    if not diff and (not module.check_mode):
        module.exit_json(msg='No changes found to be applied.')
    elif not diff and module.check_mode:
        module.exit_json(msg='No changes found to be applied.')
    elif diff and module.check_mode:
        module.exit_json(msg='Changes found to be applied.', changed=True)
    payload_dict = {'SettingType': 'Location'}
    payload_dict.update(dict(((k, v) for k, v in loc_resp.items() if k in req_dict.keys())))
    payload_dict.update(req_filter_none)
    if req_filter_none.get('RackSlot') is None:
        payload_dict.update({'RackSlot': loc_resp.get('RackSlot')})
    return payload_dict