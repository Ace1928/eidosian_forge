import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def update_vsphere_configs(config):
    available_node_types = config['available_node_types']
    head_node_type = config['head_node_type']
    head_node = available_node_types[head_node_type]
    head_node_config = head_node['node_config']
    worker_nodes = [v for k, v in available_node_types.items() if k != head_node_type]
    worker_node_configs = [worker_node['node_config'] for worker_node in worker_nodes]
    head_node_config['resources'] = head_node['resources']
    head_resource_pool = None
    if 'resource_pool' in head_node_config:
        head_resource_pool = head_node_config['resource_pool']
    for worker_node in worker_nodes:
        worker_node['node_config']['resources'] = worker_node['resources']
    for worker_node_config in worker_node_configs:
        if not worker_node_config.get('resource_pool'):
            worker_node_config['resource_pool'] = head_resource_pool
    check_and_update_frozen_vm_configs_in_provider_section(config, head_node_config, worker_node_configs)
    update_gpu_config_in_provider_section(config, head_node_config, worker_node_configs)