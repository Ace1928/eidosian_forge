import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def wait_until_ready(self):
    import ray
    if self.is_shutdown:
        raise RuntimeError('The ray cluster has been shut down or it failed to start.')
    try:
        ray.init(address=self.address)
        if self.ray_dashboard_port is not None and _wait_service_up(self.address.split(':')[0], self.ray_dashboard_port, _RAY_DASHBOARD_STARTUP_TIMEOUT):
            self.start_hook.on_ray_dashboard_created(self.ray_dashboard_port)
        else:
            try:
                __import__('ray.dashboard.optional_deps')
            except ModuleNotFoundError:
                _logger.warning('Dependencies to launch the optional dashboard API server cannot be found. They can be installed with pip install ray[default].')
        if self.autoscale:
            return
        last_alive_worker_count = 0
        last_progress_move_time = time.time()
        while True:
            time.sleep(_RAY_CLUSTER_STARTUP_PROGRESS_CHECKING_INTERVAL)
            if self.background_job_exception is not None:
                raise RuntimeError('Ray workers failed to start.') from self.background_job_exception
            cur_alive_worker_count = len([node for node in ray.nodes() if node['Alive']]) - 1
            if cur_alive_worker_count >= self.num_worker_nodes:
                return
            if cur_alive_worker_count > last_alive_worker_count:
                last_alive_worker_count = cur_alive_worker_count
                last_progress_move_time = time.time()
                _logger.info(f'Ray worker nodes are starting. Progress: ({cur_alive_worker_count} / {self.num_worker_nodes})')
            elif time.time() - last_progress_move_time > _RAY_CONNECT_CLUSTER_POLL_PROGRESS_TIMEOUT:
                if cur_alive_worker_count == 0:
                    raise RuntimeError('Current spark cluster has no resources to launch Ray worker nodes.')
                _logger.warning(f'Timeout in waiting for all ray workers to start. Started / Total requested: ({cur_alive_worker_count} / {self.num_worker_nodes}). Current spark cluster does not have sufficient resources to launch requested number of Ray worker nodes.')
                return
    finally:
        ray.shutdown()