import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
@dataclass(init=True)
class UsageStatsToReport:
    """Usage stats to report"""
    ray_version: str
    python_version: str
    schema_version: str
    source: str
    session_id: str
    git_commit: str
    os: str
    collect_timestamp_ms: int
    session_start_timestamp_ms: int
    cloud_provider: Optional[str]
    min_workers: Optional[int]
    max_workers: Optional[int]
    head_node_instance_type: Optional[str]
    worker_node_instance_types: Optional[List[str]]
    total_num_cpus: Optional[int]
    total_num_gpus: Optional[int]
    total_memory_gb: Optional[float]
    total_object_store_memory_gb: Optional[float]
    library_usages: Optional[List[str]]
    total_success: int
    total_failed: int
    seq_number: int
    extra_usage_tags: Optional[Dict[str, str]]
    total_num_nodes: Optional[int]
    total_num_running_jobs: Optional[int]
    libc_version: Optional[str]
    hardware_usages: Optional[List[str]]