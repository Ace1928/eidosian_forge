from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def update_cluster_limits(self, entity):
    new_limits = {}
    for cluster in self._module.params.get('clusters'):
        new_limits[cluster.get('name', '')] = {'cpu': int(cluster.get('cpu')), 'memory': float(cluster.get('memory'))}
    old_limits = {}
    cl_limit_service = self._service.service(entity.id).quota_cluster_limits_service()
    for limit in cl_limit_service.list():
        cluster = get_link_name(self._connection, limit.cluster) if limit.cluster else ''
        old_limits[cluster] = {'cpu': limit.vcpu_limit, 'memory': limit.memory_limit}
        cl_limit_service.service(limit.id).remove()
    return new_limits == old_limits