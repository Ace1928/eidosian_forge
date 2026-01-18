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
def update_nodes(self):
    """Run NodeUpdaterThreads to run setup commands, sync files,
        and/or start Ray.
        """
    T = []
    for node_id, setup_commands, ray_start_commands, docker_config in (self.should_update(node_id) for node_id in self.non_terminated_nodes.worker_ids):
        if node_id is not None:
            resources = self._node_resources(node_id)
            labels = self._node_labels(node_id)
            logger.debug(f'{node_id}: Starting new thread runner.')
            T.append(threading.Thread(target=self.spawn_updater, args=(node_id, setup_commands, ray_start_commands, resources, labels, docker_config)))
    for t in T:
        t.start()
    for t in T:
        t.join()