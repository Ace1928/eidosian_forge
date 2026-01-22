import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set
from ray.autoscaler._private.node_launcher import BaseNodeLauncher
from ray.autoscaler.node_provider import NodeProvider as NodeProviderV1
from ray.autoscaler.tags import TAG_RAY_USER_NODE_TYPE
from ray.autoscaler.v2.instance_manager.config import NodeProviderConfig
from ray.core.generated.instance_manager_pb2 import Instance
def _filter_instances(self, instances: Dict[str, Instance], instance_ids_filter: Set[str], instance_states_filter: Set[int]) -> Dict[str, Instance]:
    filtered = {}
    for instance_id, instance in instances.items():
        if instance_ids_filter and instance_id not in instance_ids_filter:
            continue
        if instance_states_filter and instance.state not in instance_states_filter:
            continue
        filtered[instance_id] = instance
    return filtered