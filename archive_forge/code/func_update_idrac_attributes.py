from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def update_idrac_attributes(idrac, module, uri_dict, idrac_response_attr, system_response_attr, lc_response_attr):
    resp = {}
    idrac_payload = module.params.get('idrac_attributes')
    system_payload = module.params.get('system_attributes')
    lc_payload = module.params.get('lifecycle_controller_attributes')
    if idrac_payload is not None and idrac_response_attr is not None:
        idrac_response = idrac.invoke_request(uri_dict.get(MANAGER_ID), 'PATCH', data={ATTR: idrac_payload})
        resp['iDRAC'] = idrac_response.json_data
    if system_payload is not None and system_response_attr is not None:
        system_response = idrac.invoke_request(uri_dict.get(SYSTEM_ID), 'PATCH', data={ATTR: system_payload})
        resp['System'] = system_response.json_data
    if lc_payload is not None and lc_response_attr is not None:
        lc_response = idrac.invoke_request(uri_dict.get(LC_ID), 'PATCH', data={ATTR: lc_payload})
        resp['Lifecycle Controller'] = lc_response.json_data
    return resp