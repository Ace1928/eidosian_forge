from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _update_nodes_list(self):
    """get a fresh list of all the nodes"""
    self._nodes = self._client.get_nodes()
    if not self._nodes:
        self.module.fail_json('Failed to retrieve at least 1 node.')