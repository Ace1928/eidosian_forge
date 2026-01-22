from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _has_migs(self, local):
    if local:
        return self._local_node_has_migs()
    return self._cluster_has_migs()