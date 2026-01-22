import copy
import logging
import math
import operator
import os
import queue
import subprocess
import threading
import time
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import yaml
import ray
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.legacy_info_string import legacy_log_info_string
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.local.node_provider import (
from ray.autoscaler._private.node_launcher import BaseNodeLauncher, NodeLauncher
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.node_tracker import NodeTracker
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler._private.resource_demand_scheduler import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.exceptions import RpcError
def _keep_worker_of_node_type(self, node_id: NodeID, node_type_counts: Dict[NodeType, int]) -> Tuple[KeepOrTerminate, Optional[str]]:
    """Determines if a worker should be kept based on the min_workers
        and max_workers constraint of the worker's node_type.

        Returns KeepOrTerminate.keep when both of the following hold:
        (a) The worker's node_type is present among the keys of the current
            config's available_node_types dict.
        (b) Deleting the node would violate the min_workers constraint for that
            worker's node_type.

        Returns KeepOrTerminate.terminate when both the following hold:
        (a) The worker's node_type is not present among the keys of the current
            config's available_node_types dict.
        (b) Keeping the node would violate the max_workers constraint for that
            worker's node_type.

        Return KeepOrTerminate.decide_later otherwise.

        Args:
            node_type_counts(Dict[NodeType, int]): The non_terminated node
                types counted so far.
        Returns:
            KeepOrTerminate: keep if the node should be kept, terminate if the
            node should be terminated, decide_later if we are allowed
            to terminate it, but do not have to.
            Optional[str]: reason for termination. Not None on
            KeepOrTerminate.terminate, None otherwise.
        """
    assert self.provider
    tags = self.provider.node_tags(node_id)
    if TAG_RAY_USER_NODE_TYPE in tags:
        node_type = tags[TAG_RAY_USER_NODE_TYPE]
        min_workers = self.available_node_types.get(node_type, {}).get('min_workers', 0)
        max_workers = self.available_node_types.get(node_type, {}).get('max_workers', 0)
        if node_type not in self.available_node_types:
            available_node_types = list(self.available_node_types.keys())
            return (KeepOrTerminate.terminate, f'not in available_node_types: {available_node_types}')
        new_count = node_type_counts[node_type] + 1
        if new_count <= min(min_workers, max_workers):
            return (KeepOrTerminate.keep, None)
        if new_count > max_workers:
            return (KeepOrTerminate.terminate, 'max_workers_per_type')
    return (KeepOrTerminate.decide_later, None)