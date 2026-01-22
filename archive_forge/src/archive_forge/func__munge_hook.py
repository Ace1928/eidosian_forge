from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def _munge_hook(hook_obj):
    retval = {'active': hook_obj.active, 'events': hook_obj.events, 'id': hook_obj.id, 'url': hook_obj.url}
    retval.update(hook_obj.config)
    retval['has_shared_secret'] = 'secret' in retval
    if 'secret' in retval:
        del retval['secret']
    retval['last_response'] = hook_obj.last_response.raw_data
    return retval