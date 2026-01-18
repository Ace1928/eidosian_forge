from __future__ import (absolute_import, division, print_function)
import json
import socket
import copy
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def transform_diff(params, translator, sub_payload, bool_trans=None):
    df = {}
    inp_dict = {}
    for k, v in translator.items():
        inp = params.get(k)
        if inp is not None:
            if isinstance(inp, bool) and bool_trans:
                inp = bool_trans.get(inp)
            inp_dict[v] = inp
    id_diff = recursive_diff(inp_dict, sub_payload)
    if id_diff and id_diff[0]:
        df = id_diff[0]
        sub_payload.update(inp_dict)
    return df