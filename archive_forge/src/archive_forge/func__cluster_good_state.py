from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _cluster_good_state(self):
    """checks a few things to make sure we're OK to say the cluster
        has no migs. It could be in a unhealthy condition that does not allow
        migs, or a split brain"""
    if self._cluster_key_consistent() is not True:
        return (False, 'Cluster key inconsistent.')
    if self._is_min_cluster_size() is not True:
        return (False, 'Cluster min size not reached.')
    if self._cluster_migrates_allowed() is not True:
        return (False, 'migrate_allowed is false somewhere.')
    return (True, 'OK.')