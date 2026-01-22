from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _update_cluster(self):
    cluster = self.get_cluster()
    args = self._get_common_cluster_args()
    args['id'] = cluster['id']
    if self.has_changed(args, cluster):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateCluster', **args)
            cluster = res['cluster']
    return cluster