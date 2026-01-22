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
def _start_ray_worker_nodes(*, spark, spark_job_group_id, spark_job_group_desc, num_worker_nodes, using_stage_scheduling, ray_head_ip, ray_head_port, ray_temp_dir, num_cpus_per_node, num_gpus_per_node, heap_memory_per_node, object_store_memory_per_node, worker_node_options, collect_log_to_path, autoscale_mode, spark_job_server_port):

    def ray_cluster_job_mapper(_):
        from pyspark.taskcontext import TaskContext
        _worker_logger = logging.getLogger('ray.util.spark.worker')
        context = TaskContext.get()
        worker_port_range_begin, worker_port_range_end = _preallocate_ray_worker_port_range()
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray_worker_node_dashboard_agent_port = get_random_unused_port(ray_head_ip, min_port=10000, max_port=20000)
        ray_worker_node_cmd = [sys.executable, '-m', 'ray.util.spark.start_ray_node', f'--temp-dir={ray_temp_dir}', f'--num-cpus={num_cpus_per_node}', '--block', f'--address={ray_head_ip}:{ray_head_port}', f'--memory={heap_memory_per_node}', f'--object-store-memory={object_store_memory_per_node}', f'--min-worker-port={worker_port_range_begin}', f'--max-worker-port={worker_port_range_end - 1}', f'--dashboard-agent-listen-port={ray_worker_node_dashboard_agent_port}', *_convert_ray_node_options(worker_node_options)]
        ray_worker_node_extra_envs = {RAY_ON_SPARK_COLLECT_LOG_TO_PATH: collect_log_to_path or '', RAY_ON_SPARK_START_RAY_PARENT_PID: str(os.getpid()), 'RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER': '1'}
        if num_gpus_per_node > 0:
            task_resources = context.resources()
            if 'gpu' not in task_resources:
                raise RuntimeError("Couldn't get the gpu id, Please check the GPU resource configuration")
            gpu_addr_list = [int(addr.strip()) for addr in task_resources['gpu'].addresses]
            available_physical_gpus = get_spark_task_assigned_physical_gpus(gpu_addr_list)
            ray_worker_node_cmd.append(f'--num-gpus={len(available_physical_gpus)}')
            ray_worker_node_extra_envs['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_id) for gpu_id in available_physical_gpus])
        _worker_logger.info(f'Start Ray worker, command: {' '.join(ray_worker_node_cmd)}')
        try:
            if autoscale_mode:
                requests.post(url=f'http://{ray_head_ip}:{spark_job_server_port}/notify_task_launched', json={'spark_job_group_id': spark_job_group_id})
            exec_cmd(ray_worker_node_cmd, synchronous=True, extra_env=ray_worker_node_extra_envs)
        except Exception as e:
            if autoscale_mode:
                _logger.warning(f'Ray worker node process exit, reason: {repr(e)}.')
            else:
                raise
        yield 0
    spark.sparkContext.setJobGroup(spark_job_group_id, spark_job_group_desc)
    job_rdd = spark.sparkContext.parallelize(list(range(num_worker_nodes)), num_worker_nodes)
    if using_stage_scheduling:
        resource_profile = _create_resource_profile(num_cpus_per_node, num_gpus_per_node)
        job_rdd = job_rdd.withResources(resource_profile)
    job_rdd.mapPartitions(ray_cluster_job_mapper).collect()