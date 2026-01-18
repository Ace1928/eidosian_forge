import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
@classmethod
def to_summary(cls, *, objects: List[Dict]):
    summary = {}
    total_objects = 0
    total_size_mb = 0
    key_to_workers = {}
    key_to_nodes = {}
    callsite_enabled = True
    for object in objects:
        key = object['call_site']
        if key == 'disabled':
            callsite_enabled = False
        if key not in summary:
            summary[key] = ObjectSummaryPerKey(total_objects=0, total_size_mb=0, total_num_workers=0, total_num_nodes=0)
            key_to_workers[key] = set()
            key_to_nodes[key] = set()
        object_summary = summary[key]
        task_state = object['task_status']
        if task_state not in object_summary.task_state_counts:
            object_summary.task_state_counts[task_state] = 0
        object_summary.task_state_counts[task_state] += 1
        ref_type = object['reference_type']
        if ref_type not in object_summary.ref_type_counts:
            object_summary.ref_type_counts[ref_type] = 0
        object_summary.ref_type_counts[ref_type] += 1
        object_summary.total_objects += 1
        total_objects += 1
        size_bytes = object['object_size']
        if size_bytes != -1:
            object_summary.total_size_mb += size_bytes / 1024 ** 2
            total_size_mb += size_bytes / 1024 ** 2
        key_to_workers[key].add(object['pid'])
        key_to_nodes[key].add(object['ip'])
    for key, workers in key_to_workers.items():
        summary[key].total_num_workers = len(workers)
    for key, nodes in key_to_nodes.items():
        summary[key].total_num_nodes = len(nodes)
    return ObjectSummaries(summary=summary, total_objects=total_objects, total_size_mb=total_size_mb, callsite_enabled=callsite_enabled)