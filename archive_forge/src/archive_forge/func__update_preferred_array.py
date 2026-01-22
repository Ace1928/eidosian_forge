from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _update_preferred_array(module, array, answer=False):
    """Update existing preferred array list. Only called when supported"""
    preferred_array = array.get_host(module.params['name'], preferred_array=True)['preferred_array']
    if preferred_array == [] and module.params['preferred_array'] != ['delete']:
        answer = True
        if not module.check_mode:
            try:
                array.set_host(module.params['name'], preferred_array=module.params['preferred_array'])
            except Exception:
                module.fail_json(msg='Preferred array list creation failed for {0}.'.format(module.params['name']))
    elif preferred_array != []:
        if module.params['preferred_array'] == ['delete']:
            answer = True
            if not module.check_mode:
                try:
                    array.set_host(module.params['name'], preferred_array=[])
                except Exception:
                    module.fail_json(msg='Preferred array list deletion failed for {0}.'.format(module.params['name']))
        elif preferred_array != module.params['preferred_array']:
            answer = True
            if not module.check_mode:
                try:
                    array.set_host(module.params['name'], preferred_array=module.params['preferred_array'])
                except Exception:
                    module.fail_json(msg='Preferred array list change failed for {0}.'.format(module.params['name']))
    return answer