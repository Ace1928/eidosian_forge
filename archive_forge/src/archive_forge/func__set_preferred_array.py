from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _set_preferred_array(module, array):
    """Set preferred array list. Only called when supported"""
    if module.params['preferred_array'] != ['delete']:
        array.set_host(module.params['name'], preferred_array=module.params['preferred_array'])
    else:
        array.set_host(module.params['name'], preferred_array=[])