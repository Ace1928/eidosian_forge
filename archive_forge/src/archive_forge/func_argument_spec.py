from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import env_fallback
def argument_spec():
    """Return standard base dictionary used for the argument_spec argument in AnsibleModule"""
    return dict(array=dict(type='str', required=True), user=dict(type='str', fallback=(env_fallback, ['VEXATA_USER'])), password=dict(type='str', no_log=True, fallback=(env_fallback, ['VEXATA_PASSWORD'])), validate_certs=dict(type='bool', required=False, default=False))