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
def _get_all_stats(self):
    now = dashboard_utils.to_posix_time(datetime.datetime.utcnow())
    network_stats = self._get_network_stats()
    self._network_stats_hist.append((now, network_stats))
    network_speed_stats = self._compute_speed_from_hist(self._network_stats_hist)
    disk_stats = self._get_disk_io_stats()
    self._disk_io_stats_hist.append((now, disk_stats))
    disk_speed_stats = self._compute_speed_from_hist(self._disk_io_stats_hist)
    return {'now': now, 'hostname': self._hostname, 'ip': self._ip, 'cpu': self._get_cpu_percent(IN_KUBERNETES_POD), 'cpus': self._cpu_counts, 'mem': self._get_mem_usage(), 'shm': self._get_shm_usage(), 'workers': self._get_workers(), 'raylet': self._get_raylet(), 'agent': self._get_agent(), 'bootTime': self._get_boot_time(), 'loadAvg': self._get_load_avg(), 'disk': self._get_disk_usage(), 'disk_io': disk_stats, 'disk_io_speed': disk_speed_stats, 'gpus': self._get_gpu_usage(), 'network': network_stats, 'network_speed': network_speed_stats, 'cmdline': self._get_raylet().get('cmdline', [])}