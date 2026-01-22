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
class NonTerminatedNodes:
    """Class to extract and organize information on non-terminated nodes."""

    def __init__(self, provider: NodeProvider):
        start_time = time.time()
        self.all_node_ids = provider.non_terminated_nodes({})
        self.worker_ids: List[NodeID] = []
        self.head_id: Optional[NodeID] = None
        for node in self.all_node_ids:
            node_kind = provider.node_tags(node)[TAG_RAY_NODE_KIND]
            if node_kind == NODE_KIND_WORKER:
                self.worker_ids.append(node)
            elif node_kind == NODE_KIND_HEAD:
                self.head_id = node
        self.non_terminated_nodes_time = time.time() - start_time
        logger.info(f'The autoscaler took {round(self.non_terminated_nodes_time, 3)} seconds to fetch the list of non-terminated nodes.')

    def remove_terminating_nodes(self, terminating_nodes: List[NodeID]) -> None:
        """Remove nodes we're in the process of terminating from internal
        state."""

        def not_terminating(node):
            return node not in terminating_nodes
        self.worker_ids = list(filter(not_terminating, self.worker_ids))
        self.all_node_ids = list(filter(not_terminating, self.all_node_ids))