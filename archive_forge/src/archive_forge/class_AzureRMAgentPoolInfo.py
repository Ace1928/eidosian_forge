from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAgentPoolInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), cluster_name=dict(type='str', required=True))
        self.results = dict()
        self.resource_group = None
        self.name = None
        self.cluster_name = None
        super(AzureRMAgentPoolInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec):
            setattr(self, key, kwargs[key])
        if self.name:
            aks_agent_pools = [self.get_agentpool()]
        else:
            aks_agent_pools = self.list_agentpool()
        self.results['aks_agent_pools'] = [self.to_dict(x) for x in aks_agent_pools]
        return self.results

    def get_agentpool(self):
        try:
            return self.managedcluster_client.agent_pools.get(self.resource_group, self.cluster_name, self.name)
        except ResourceNotFoundError:
            pass

    def list_agentpool(self):
        result = []
        try:
            resp = self.managedcluster_client.agent_pools.list(self.resource_group, self.cluster_name)
            while True:
                result.append(resp.next())
        except StopIteration:
            pass
        except Exception:
            pass
        return result

    def to_dict(self, agent_pool):
        if not agent_pool:
            return None
        agent_pool_dict = dict(resource_group=self.resource_group, cluster_name=self.cluster_name, id=agent_pool.id, type=agent_pool.type, name=agent_pool.name, count=agent_pool.count, vm_size=agent_pool.vm_size, os_disk_size_gb=agent_pool.os_disk_size_gb, vnet_subnet_id=agent_pool.vnet_subnet_id, max_pods=agent_pool.max_pods, os_type=agent_pool.os_type, max_count=agent_pool.max_count, min_count=agent_pool.min_count, enable_auto_scaling=agent_pool.enable_auto_scaling, type_properties_type=agent_pool.type_properties_type, mode=agent_pool.mode, availability_zones=[], orchestrator_version=agent_pool.orchestrator_version, node_image_version=agent_pool.node_image_version, upgrade_settings=dict(), provisioning_state=agent_pool.provisioning_state, enable_node_public_ip=agent_pool.enable_node_public_ip, scale_set_priority=agent_pool.scale_set_priority, scale_set_eviction_policy=agent_pool.scale_set_eviction_policy, spot_max_price=agent_pool.spot_max_price, node_labels=agent_pool.node_labels, node_taints=agent_pool.node_taints)
        if agent_pool.upgrade_settings is not None:
            agent_pool_dict['upgrade_settings']['max_surge'] = agent_pool.upgrade_settings.max_surge
        if agent_pool.availability_zones is not None:
            for key in agent_pool.availability_zones:
                agent_pool_dict['availability_zones'].append(int(key))
        return agent_pool_dict