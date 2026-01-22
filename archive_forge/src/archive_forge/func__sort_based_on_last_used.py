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
def _sort_based_on_last_used(self, nodes: List[NodeID], last_used: Dict[str, float]) -> List[NodeID]:
    """Sort the nodes based on the last time they were used.

        The first item in the return list is the most recently used.
        """
    last_used_copy = copy.deepcopy(last_used)
    least_recently_used = -1

    def last_time_used(node_id: NodeID):
        assert self.provider
        node_ip = self.provider.internal_ip(node_id)
        if node_ip not in last_used_copy:
            return least_recently_used
        else:
            return last_used_copy[node_ip]
    return sorted(nodes, key=last_time_used, reverse=True)