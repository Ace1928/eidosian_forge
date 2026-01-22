from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAksAgentPool(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), cluster_name=dict(type='str', required=True), count=dict(type='int'), vm_size=dict(type='str'), os_disk_size_gb=dict(type='int'), vnet_subnet_id=dict(type='str'), availability_zones=dict(type='list', elements='int', choices=[1, 2, 3]), os_type=dict(type='str', choices=['Linux', 'Windows']), orchestrator_version=dict(type='str'), type_properties_type=dict(type='str', choices=['VirtualMachineScaleSets', 'AvailabilitySet']), mode=dict(type='str', choices=['System', 'User']), enable_auto_scaling=dict(type='bool'), max_count=dict(type='int'), node_labels=dict(type='dict'), min_count=dict(type='int'), max_pods=dict(type='int'), state=dict(type='str', choices=['present', 'absent'], default='present'))
        self.results = dict()
        self.resource_group = None
        self.name = None
        self.cluster_name = None
        self.count = None
        self.vm_size = None
        self.mode = None
        self.os_disk_size_gb = None
        self.storage_profiles = None
        self.vnet_subnet_id = None
        self.availability_zones = None
        self.os_type = None
        self.orchestrator_version = None
        self.type_properties_type = None
        self.enable_auto_scaling = None
        self.max_count = None
        self.node_labels = None
        self.min_count = None
        self.max_pods = None
        self.body = dict()
        super(AzureRMAksAgentPool, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec):
            setattr(self, key, kwargs[key])
            if key not in ['resource_group', 'cluster_name', 'name', 'state']:
                self.body[key] = kwargs[key]
        agent_pool = self.get()
        changed = False
        response = None
        if self.state == 'present':
            if agent_pool:
                for key in self.body.keys():
                    if self.body[key] is not None and self.body[key] != agent_pool[key]:
                        changed = True
                    else:
                        self.body[key] = agent_pool[key]
            else:
                changed = True
            if changed:
                if not self.check_mode:
                    response = self.create_or_update(self.body)
        elif not self.check_mode:
            if agent_pool:
                response = self.delete_agentpool()
                changed = True
            else:
                changed = False
        else:
            changed = True
        self.results['changed'] = changed
        self.results['aks_agent_pools'] = response
        return self.results

    def get(self):
        try:
            response = self.managedcluster_client.agent_pools.get(self.resource_group, self.cluster_name, self.name)
            return self.to_dict(response)
        except ResourceNotFoundError:
            pass

    def create_or_update(self, parameters):
        try:
            response = self.managedcluster_client.agent_pools.begin_create_or_update(self.resource_group, self.cluster_name, self.name, parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return self.to_dict(response)
        except Exception as exc:
            self.fail('Error when creating cluster node agent pool {0}: {1}'.format(self.name, exc))

    def delete_agentpool(self):
        try:
            response = self.managedcluster_client.agent_pools.begin_delete(self.resource_group, self.cluster_name, self.name)
        except Exception as exc:
            self.fail('Error when deleting cluster agent pool {0}: {1}'.format(self.name, exc))

    def to_dict(self, agent_pool):
        if not agent_pool:
            return None
        agent_pool_dict = dict(resource_group=self.resource_group, cluster_name=self.cluster_name, id=agent_pool.id, type=agent_pool.type, name=agent_pool.name, count=agent_pool.count, vm_size=agent_pool.vm_size, os_disk_size_gb=agent_pool.os_disk_size_gb, vnet_subnet_id=agent_pool.vnet_subnet_id, max_pods=agent_pool.max_pods, os_type=agent_pool.os_type, max_count=agent_pool.max_count, min_count=agent_pool.min_count, enable_auto_scaling=agent_pool.enable_auto_scaling, type_properties_type=agent_pool.type_properties_type, mode=agent_pool.mode, orchestrator_version=agent_pool.orchestrator_version, node_image_version=agent_pool.node_image_version, upgrade_settings=dict(), provisioning_state=agent_pool.provisioning_state, availability_zones=[], enable_node_public_ip=agent_pool.enable_node_public_ip, scale_set_priority=agent_pool.scale_set_priority, scale_set_eviction_policy=agent_pool.scale_set_eviction_policy, spot_max_price=agent_pool.spot_max_price, node_labels=agent_pool.node_labels, node_taints=agent_pool.node_taints)
        if agent_pool.upgrade_settings is not None:
            agent_pool_dict['upgrade_settings']['max_surge'] = agent_pool.upgrade_settings.max_surge
        if agent_pool.availability_zones is not None:
            for key in agent_pool.availability_zones:
                agent_pool_dict['availability_zones'].append(int(key))
        return agent_pool_dict