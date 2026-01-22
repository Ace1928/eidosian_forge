from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _cluster_key_consistent(self):
    """create a dictionary to store what each node
        returns the cluster key as. we should end up with only 1 dict key,
        with the key being the cluster key."""
    cluster_keys = {}
    for node in self._nodes:
        cluster_key = self._cluster_statistics[node]['cluster_key']
        if cluster_key not in cluster_keys:
            cluster_keys[cluster_key] = 1
        else:
            cluster_keys[cluster_key] += 1
    if len(cluster_keys.keys()) == 1 and self._start_cluster_key in cluster_keys:
        return True
    return False