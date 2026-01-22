from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class QuotasModule(BaseModule):

    def build_entity(self):
        return otypes.Quota(description=self._module.params['description'], name=self._module.params['name'], id=self._module.params['id'], storage_hard_limit_pct=self._module.params.get('storage_grace'), storage_soft_limit_pct=self._module.params.get('storage_threshold'), cluster_hard_limit_pct=self._module.params.get('cluster_grace'), cluster_soft_limit_pct=self._module.params.get('cluster_threshold'))

    def update_storage_limits(self, entity):
        new_limits = {}
        for storage in self._module.params.get('storages'):
            new_limits[storage.get('name', '')] = {'size': storage.get('size')}
        old_limits = {}
        sd_limit_service = self._service.service(entity.id).quota_storage_limits_service()
        for limit in sd_limit_service.list():
            storage = get_link_name(self._connection, limit.storage_domain) if limit.storage_domain else ''
            old_limits[storage] = {'size': limit.limit}
            sd_limit_service.service(limit.id).remove()
        return new_limits == old_limits

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

    def update_check(self, entity):
        return self.update_storage_limits(entity) and self.update_cluster_limits(entity) and equal(self._module.params.get('name'), entity.name) and equal(self._module.params.get('description'), entity.description) and equal(self._module.params.get('storage_grace'), entity.storage_hard_limit_pct) and equal(self._module.params.get('storage_threshold'), entity.storage_soft_limit_pct) and equal(self._module.params.get('cluster_grace'), entity.cluster_hard_limit_pct) and equal(self._module.params.get('cluster_threshold'), entity.cluster_soft_limit_pct)