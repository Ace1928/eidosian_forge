import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ray._private.protobuf_compat import message_to_dict
from ray.autoscaler._private.resource_demand_scheduler import UtilizationScore
from ray.autoscaler.v2.schema import NodeType
from ray.autoscaler.v2.utils import is_pending, resource_requests_by_count
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
def _enforce_min_workers(self) -> None:
    """
        Enforce the minimal count of nodes for each worker node type.
        """
    count_by_node_type = self._ctx.get_cluster_shape()
    logger.debug('Enforcing min workers: {}'.format(self._ctx))
    new_nodes = []
    for node_type, node_type_config in self._ctx.get_cluster_config().node_type_configs.items():
        cur_count = count_by_node_type.get(node_type, 0)
        min_count = node_type_config.min_workers
        if cur_count < min_count:
            new_nodes.extend([SchedulingNode.from_node_config(copy.deepcopy(node_type_config), status=SchedulingNodeStatus.TO_LAUNCH)] * (min_count - cur_count))
    self._ctx.update(new_nodes + self._ctx.get_nodes())
    logger.debug('After enforced min workers: {}'.format(self._ctx))