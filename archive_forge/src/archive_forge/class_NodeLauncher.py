import copy
import logging
import operator
import threading
import time
import traceback
from typing import Any, Dict, Optional
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import hash_launch_conf
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.tags import (
class NodeLauncher(BaseNodeLauncher, threading.Thread):
    """Launches nodes asynchronously in the background."""

    def __init__(self, provider, queue, pending, event_summarizer, node_provider_availability_tracker, session_name: Optional[str]=None, prom_metrics=None, node_types=None, index=None, *thread_args, **thread_kwargs):
        self.queue = queue
        BaseNodeLauncher.__init__(self, provider=provider, pending=pending, event_summarizer=event_summarizer, session_name=session_name, node_provider_availability_tracker=node_provider_availability_tracker, prom_metrics=prom_metrics, node_types=node_types, index=index)
        threading.Thread.__init__(self, *thread_args, **thread_kwargs)

    def run(self):
        """Collects launch data from queue populated by StandardAutoscaler.
        Launches nodes in a background thread.

        Overrides threading.Thread.run().
        NodeLauncher.start() executes this loop in a background thread.
        """
        while True:
            config, count, node_type = self.queue.get()
            self.launch_node(config, count, node_type)