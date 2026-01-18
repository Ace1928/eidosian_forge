from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def validate_vlans(module, rest_obj):
    vlan_name_id_map = get_vlan_name_id_map(rest_obj)
    vlan_name_id_map['0'] = 0
    tagged_list = module.params.get('tagged_networks')
    untag_list = module.params.get('untagged_networks')
    untag_dict = {}
    if untag_list:
        for utg in untag_list:
            p = utg['port']
            if utg.get('untagged_network_id') is not None:
                if p in untag_dict:
                    module.fail_json(msg='port {0} is repeated for untagged_network_id'.format(p))
                vlan = utg.get('untagged_network_id')
                if vlan not in vlan_name_id_map.values():
                    module.fail_json(msg='untagged_network_id: {0} is not a valid vlan id for port {1}'.format(vlan, p))
                untag_dict[p] = vlan
            if utg.get('untagged_network_name'):
                vlan = utg.get('untagged_network_name')
                if vlan in vlan_name_id_map:
                    if p in untag_dict:
                        module.fail_json(msg='port {0} is repeated for untagged_network_name'.format(p))
                    untag_dict[p] = vlan_name_id_map.get(vlan)
                else:
                    module.fail_json(msg='{0} is not a valid vlan name for port {1}'.format(vlan, p))
    vlan_name_id_map.pop('0')
    tagged_dict = {}
    if tagged_list:
        for tg in tagged_list:
            p = tg['port']
            tg_list = []
            empty_list = False
            tgnids = tg.get('tagged_network_ids')
            if isinstance(tgnids, list):
                if len(tgnids) == 0:
                    empty_list = True
                for vl in tgnids:
                    if vl not in vlan_name_id_map.values():
                        module.fail_json(msg='{0} is not a valid vlan id port {1}'.format(vl, p))
                    tg_list.append(vl)
            tgnames = tg.get('tagged_network_names')
            if isinstance(tgnames, list):
                if len(tgnames) == 0:
                    empty_list = True
                for vln in tgnames:
                    if vln not in vlan_name_id_map:
                        module.fail_json(msg='{0} is not a valid vlan name port {1}'.format(vln, p))
                    tg_list.append(vlan_name_id_map.get(vln))
            if not tg_list and (not empty_list):
                module.fail_json(msg='No tagged_networks provided or valid tagged_networks not found for port {0}'.format(p))
            tagged_dict[p] = list(set(tg_list))
    for k, v in untag_dict.items():
        if v in tagged_dict.get(k, []):
            module.fail_json(msg="vlan {0}('{1}') cannot be in both tagged and untagged list for port {2}".format(v, get_key(v, vlan_name_id_map), k))
    return (untag_dict, tagged_dict)