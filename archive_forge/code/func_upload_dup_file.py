from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def upload_dup_file(rest_obj, module):
    """Upload DUP file to OME and get a file token."""
    upload_uri = 'UpdateService/Actions/UpdateService.UploadFile'
    headers = {'Content-Type': 'application/octet-stream', 'Accept': 'application/octet-stream'}
    upload_success, token = (False, None)
    dup_file = module.params['dup_file']
    with open(dup_file, 'rb') as payload:
        payload = payload.read()
        response = rest_obj.invoke_request('POST', upload_uri, data=payload, headers=headers, api_timeout=100, dump=False)
        if response.status_code == 200:
            upload_success = True
            token = str(response.json_data)
        else:
            module.fail_json(msg='Unable to upload {0} to {1}'.format(dup_file, module.params['hostname']))
    return (upload_success, token)