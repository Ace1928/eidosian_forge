from __future__ import (absolute_import, division, print_function)
import json
import base64
import os
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import reset_idrac
def _build_import_payload(module, cert_type):
    payload = {'CertificateType': cert_type}
    if module.params.get('passphrase'):
        payload['Passphrase'] = module.params.get('passphrase')
    cert_path = module.params.get('certificate_path')
    try:
        if str(cert_path).lower().endswith('.p12') or str(cert_path).lower().endswith('.pfx'):
            with open(cert_path, 'rb') as cert_file:
                cert_content = cert_file.read()
                cert_file_content = base64.encodebytes(cert_content).decode('ascii')
        else:
            with open(cert_path, 'r') as cert_file:
                cert_file_content = cert_file.read()
    except OSError as file_error:
        module.exit_json(msg=str(file_error), failed=True)
    payload['SSLCertificateFile'] = cert_file_content
    return payload