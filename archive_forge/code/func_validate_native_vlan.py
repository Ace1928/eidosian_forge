from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_item_and_list
def validate_native_vlan(module, rest_obj, fabric_id, media_id):
    resp = rest_obj.invoke_request('POST', APPLICABLE_UNTAGGED.format(fabric_id=fabric_id), data={'UplinkType': media_id})
    vlans = []
    if resp.json_data.get('ApplicableUplinkNetworks'):
        vlans = resp.json_data.get('ApplicableUplinkNetworks', [])
    vlan_id = 0
    vlan_name = module.params.get('untagged_network')
    for vlan in vlans:
        if vlan['Name'] == vlan_name:
            vlan_id = vlan['VlanMaximum']
            break
    if not vlan_id:
        module.fail_json(msg='Native VLAN name {0} is not applicable or valid.'.format(vlan_name))
    return vlan_id