import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
from ray.autoscaler.tags import (
def _scale_request_to_patch_payload(self, scale_request: ScaleRequest, raycluster: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converts autoscaler scale request into a RayCluster CR patch payload."""
    patch_payload = []
    for node_type, target_replicas in scale_request.desired_num_workers.items():
        group_index = _worker_group_index(raycluster, node_type)
        group_max_replicas = _worker_group_max_replicas(raycluster, group_index)
        if group_max_replicas is not None and group_max_replicas < target_replicas:
            logger.warning('Autoscaler attempted to create ' + 'more than maxReplicas pods of type {}.'.format(node_type))
            target_replicas = group_max_replicas
        if target_replicas == _worker_group_replicas(raycluster, group_index):
            continue
        patch = worker_replica_patch(group_index, target_replicas)
        patch_payload.append(patch)
    deletion_groups = defaultdict(list)
    for worker in scale_request.workers_to_delete:
        node_type = self.node_tags(worker)[TAG_RAY_USER_NODE_TYPE]
        deletion_groups[node_type].append(worker)
    for node_type, workers_to_delete in deletion_groups.items():
        group_index = _worker_group_index(raycluster, node_type)
        patch = worker_delete_patch(group_index, workers_to_delete)
        patch_payload.append(patch)
    return patch_payload