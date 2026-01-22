from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _update_host_personality(module, array, answer=False):
    """Change host personality. Only called when supported"""
    personality = array.get_host(module.params['name'], personality=True)['personality']
    if personality is None and module.params['personality'] != 'delete':
        answer = True
        if not module.check_mode:
            try:
                array.set_host(module.params['name'], personality=module.params['personality'])
            except Exception:
                module.fail_json(msg='Personality setting failed.')
    if personality is not None:
        if module.params['personality'] == 'delete':
            answer = True
            if not module.check_mode:
                try:
                    array.set_host(module.params['name'], personality='')
                except Exception:
                    module.fail_json(msg='Personality deletion failed.')
        elif personality != module.params['personality']:
            answer = True
            if not module.check_mode:
                try:
                    array.set_host(module.params['name'], personality=module.params['personality'])
                except Exception:
                    module.fail_json(msg='Personality change failed.')
    return answer