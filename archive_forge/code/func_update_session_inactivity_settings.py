from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def update_session_inactivity_settings(rest_obj, payload):
    final_resp = rest_obj.invoke_request('POST', SESSION_INACTIVITY_POST, data=payload)
    return final_resp