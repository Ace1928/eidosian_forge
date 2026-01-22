from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def _create_hook_config(module):
    hook_config = {'url': module.params['url'], 'content_type': module.params['content_type'], 'insecure_ssl': '1' if module.params['insecure_ssl'] else '0'}
    secret = module.params.get('secret')
    if secret:
        hook_config['secret'] = secret
    return hook_config