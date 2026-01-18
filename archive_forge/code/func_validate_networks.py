from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_item_and_list
def validate_networks(module, rest_obj, fabric_id, media_id):
    resp = rest_obj.invoke_request('POST', APPLICABLE_NETWORKS.format(fabric_id=fabric_id), data={'UplinkType': media_id})
    vlans = []
    if resp.json_data.get('ApplicableUplinkNetworks'):
        vlans = resp.json_data.get('ApplicableUplinkNetworks', [])
    vlan_payload = []
    vlan_dict = {}
    for vlan in vlans:
        vlan_dict[vlan['Name']] = vlan['Id']
    networks = list((str(net).strip() for net in module.params.get('tagged_networks')))
    invalids = []
    for ntw in networks:
        if vlan_dict.get(ntw):
            vlan_payload.append(vlan_dict.get(ntw))
        else:
            invalids.append(ntw)
    if invalids:
        module.fail_json(msg='Networks with names {0} are not applicable or valid.'.format(','.join(set(invalids))))
    return vlan_payload