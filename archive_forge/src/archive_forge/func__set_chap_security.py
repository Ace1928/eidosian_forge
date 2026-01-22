from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _set_chap_security(module, array):
    """Set CHAP usernames and passwords"""
    pattern = re.compile('[^ ]{12,255}')
    if module.params['host_user']:
        if not pattern.match(module.params['host_password']):
            module.fail_json(msg='host_password must contain a minimum of 12 and a maximum of 255 characters')
        try:
            array.set_host(module.params['name'], host_user=module.params['host_user'], host_password=module.params['host_password'])
        except Exception:
            module.fail_json(msg='Failed to set CHAP host username and password')
    if module.params['target_user']:
        if not pattern.match(module.params['target_password']):
            module.fail_json(msg='target_password must contain a minimum of 12 and a maximum of 255 characters')
        try:
            array.set_host(module.params['name'], target_user=module.params['target_user'], target_password=module.params['target_password'])
        except Exception:
            module.fail_json(msg='Failed to set CHAP target username and password')