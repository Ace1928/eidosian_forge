from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from datetime import datetime
def _check_target(module, array):
    try:
        target = list(array.get_array_connections(names=[module.params['offload']]).items)[0]
        if target.status == 'connected':
            return True
        return False
    except Exception:
        return False