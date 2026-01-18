from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
@staticmethod
def vmware_client_argument_spec():
    return dict(hostname=dict(type='str', fallback=(env_fallback, ['VMWARE_HOST'])), username=dict(type='str', fallback=(env_fallback, ['VMWARE_USER']), aliases=['user', 'admin']), password=dict(type='str', fallback=(env_fallback, ['VMWARE_PASSWORD']), aliases=['pass', 'pwd'], no_log=True), port=dict(type='int', default=443, fallback=(env_fallback, ['VMWARE_PORT'])), protocol=dict(type='str', default='https', choices=['https', 'http']), validate_certs=dict(type='bool', fallback=(env_fallback, ['VMWARE_VALIDATE_CERTS']), default=True), proxy_host=dict(type='str', required=False, default=None, fallback=(env_fallback, ['VMWARE_PROXY_HOST'])), proxy_port=dict(type='int', required=False, default=None, fallback=(env_fallback, ['VMWARE_PROXY_PORT'])))