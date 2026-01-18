from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def validate_window(window, module):
    if window is not None:
        if 'state' not in window.keys():
            module.fail_json(msg='Balancing window state must be specified')
        elif window['state'] not in ['present', 'absent']:
            module.fail_json(msg='Balancing window state must be present or absent')
        elif window['state'] == 'present' and ('start' not in window.keys() or 'stop' not in window.keys()):
            module.fail_json(msg='Balancing window start and stop values must be specified')
    return True