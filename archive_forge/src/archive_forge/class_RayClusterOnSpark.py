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
class RayClusterOnSpark:
    """
    This class is the type of instance returned by the `_setup_ray_cluster` interface.
    Its main functionality is to:
    Connect to, disconnect from, and shutdown the Ray cluster running on Apache Spark.
    Serve as a Python context manager for the `RayClusterOnSpark` instance.

    Args
        address: The url for the ray head node (defined as the hostname and unused
                 port on Spark driver node)
        head_proc: Ray head process
        spark_job_group_id: The Spark job id for a submitted ray job
        num_workers_node: The number of workers in the ray cluster.
    """

    def __init__(self, autoscale, address, head_proc, spark_job_group_id, num_workers_node, temp_dir, cluster_unique_id, start_hook, ray_dashboard_port, spark_job_server, ray_client_server_port):
        self.autoscale = autoscale
        self.address = address
        self.head_proc = head_proc
        self.spark_job_group_id = spark_job_group_id
        self.num_worker_nodes = num_workers_node
        self.temp_dir = temp_dir
        self.cluster_unique_id = cluster_unique_id
        self.start_hook = start_hook
        self.ray_dashboard_port = ray_dashboard_port
        self.spark_job_server = spark_job_server
        self.ray_client_server_port = ray_client_server_port
        self.is_shutdown = False
        self.spark_job_is_canceled = False
        self.background_job_exception = None
        self.ray_ctx = None

    def _cancel_background_spark_job(self):
        self.spark_job_is_canceled = True
        get_spark_session().sparkContext.cancelJobGroup(self.spark_job_group_id)

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

    def connect(self):
        if ray.is_initialized():
            raise RuntimeError('Already connected to Ray cluster.')
        self.ray_ctx = ray.init(address=self.address)

    def disconnect(self):
        ray.shutdown()
        self.ray_ctx = None

    def shutdown(self, cancel_background_job=True):
        """
        Shutdown the ray cluster created by the `setup_ray_cluster` API.
        NB: In the background thread that runs the background spark job, if spark job
        raise unexpected error, its exception handler will also call this method, in
        the case, it will set cancel_background_job=False to avoid recursive call.
        """
        if not self.is_shutdown:
            self.disconnect()
            os.environ.pop('RAY_ADDRESS', None)
            if self.autoscale:
                self.spark_job_server.shutdown()
            if cancel_background_job:
                if self.autoscale:
                    pass
                else:
                    try:
                        self._cancel_background_spark_job()
                    except Exception as e:
                        _logger.warning(f'An error occurred while cancelling the ray cluster background spark job: {repr(e)}')
            try:
                self.head_proc.terminate()
            except Exception as e:
                _logger.warning(f'An Error occurred during shutdown of ray head node: {repr(e)}')
            self.is_shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()