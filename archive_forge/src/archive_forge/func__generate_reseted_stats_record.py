import asyncio
import datetime
import json
import logging
import os
import socket
import sys
import traceback
import warnings
import psutil
from typing import List, Optional, Tuple
from collections import defaultdict
import ray
import ray._private.services
import ray._private.utils
from ray.dashboard.consts import (
from ray.dashboard.modules.reporter.profile_manager import CpuProfilingManager
import ray.dashboard.modules.reporter.reporter_consts as reporter_consts
import ray.dashboard.utils as dashboard_utils
from opencensus.stats import stats as stats_module
import ray._private.prometheus_exporter as prometheus_exporter
from prometheus_client.core import REGISTRY
from ray._private.metrics_agent import Gauge, MetricsAgent, Record
from ray._private.ray_constants import DEBUG_AUTOSCALING_STATUS
from ray.core.generated import reporter_pb2, reporter_pb2_grpc
from ray.util.debug import log_once
from ray.dashboard import k8s_utils
from ray._raylet import WorkerID
def _generate_reseted_stats_record(self, component_name: str) -> List[Record]:
    """Return a list of Record that will reset
        the system metrics of a given component name.

        Args:
            component_name: a component name for a given stats.

        Returns:
            a list of Record instances of all values 0.
        """
    tags = {'ip': self._ip, 'Component': component_name}
    records = []
    records.append(Record(gauge=METRICS_GAUGES['component_cpu_percentage'], value=0.0, tags=tags))
    records.append(Record(gauge=METRICS_GAUGES['component_mem_shared_bytes'], value=0.0, tags=tags))
    records.append(Record(gauge=METRICS_GAUGES['component_rss_mb'], value=0.0, tags=tags))
    records.append(Record(gauge=METRICS_GAUGES['component_uss_mb'], value=0.0, tags=tags))
    records.append(Record(gauge=METRICS_GAUGES['component_num_fds'], value=0, tags=tags))
    return records