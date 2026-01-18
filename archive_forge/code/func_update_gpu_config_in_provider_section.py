import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def update_gpu_config_in_provider_section(config, head_node_config, worker_node_configs):
    provider_config = config['provider']
    vsphere_config = provider_config['vsphere_config']
    if 'gpu_config' in vsphere_config:
        head_node_config['gpu_config'] = vsphere_config['gpu_config']
        for worker_node_config in worker_node_configs:
            worker_node_config['gpu_config'] = vsphere_config['gpu_config']