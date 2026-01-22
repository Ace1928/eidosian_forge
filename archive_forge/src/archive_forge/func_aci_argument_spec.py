from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def aci_argument_spec():
    return dict(host=dict(type='str', aliases=['hostname'], fallback=(env_fallback, ['ACI_HOST'])), port=dict(type='int', required=False, fallback=(env_fallback, ['ACI_PORT'])), username=dict(type='str', aliases=['user'], fallback=(env_fallback, ['ACI_USERNAME', 'ANSIBLE_NET_USERNAME'])), password=dict(type='str', no_log=True, fallback=(env_fallback, ['ACI_PASSWORD', 'ANSIBLE_NET_PASSWORD'])), private_key=dict(type='str', aliases=['cert_key'], no_log=True, fallback=(env_fallback, ['ACI_PRIVATE_KEY', 'ANSIBLE_NET_SSH_KEYFILE'])), certificate_name=dict(type='str', aliases=['cert_name'], fallback=(env_fallback, ['ACI_CERTIFICATE_NAME'])), output_level=dict(type='str', default='normal', choices=['debug', 'info', 'normal'], fallback=(env_fallback, ['ACI_OUTPUT_LEVEL'])), timeout=dict(type='int', fallback=(env_fallback, ['ACI_TIMEOUT'])), use_proxy=dict(type='bool', fallback=(env_fallback, ['ACI_USE_PROXY'])), use_ssl=dict(type='bool', fallback=(env_fallback, ['ACI_USE_SSL'])), validate_certs=dict(type='bool', fallback=(env_fallback, ['ACI_VALIDATE_CERTS'])), output_path=dict(type='str', fallback=(env_fallback, ['ACI_OUTPUT_PATH'])))