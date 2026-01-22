from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _cluster_stable(self):
    """Added 4.3:
        cluster-stable:size=<target-cluster-size>;ignore-migrations=<yes/no>;namespace=<namespace-name>
        Returns the current 'cluster_key' when the following are satisfied:

         If 'size' is specified then the target node's 'cluster-size'
         must match size.
         If 'ignore-migrations' is either unspecified or 'false' then
         the target node's migrations counts must be zero for the provided
         'namespace' or all namespaces if 'namespace' is not provided."""
    cluster_key = set()
    cluster_key.add(self._info_cmd_helper('statistics')['cluster_key'])
    cmd = 'cluster-stable:'
    target_cluster_size = self.module.params['target_cluster_size']
    if target_cluster_size is not None:
        cmd = cmd + 'size=' + str(target_cluster_size) + ';'
    for node in self._nodes:
        try:
            cluster_key.add(self._info_cmd_helper(cmd, node))
        except aerospike.exception.ServerError as e:
            if 'unstable-cluster' in e.msg:
                return False
            raise e
    if len(cluster_key) == 1:
        return True
    return False