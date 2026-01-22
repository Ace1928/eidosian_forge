from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from datetime import datetime
def _check_offload(module, array):
    try:
        offload = list(array.get_offloads(names=[module.params['offload']]).items)[0]
        if offload.status == 'connected':
            return True
        return False
    except Exception:
        return False