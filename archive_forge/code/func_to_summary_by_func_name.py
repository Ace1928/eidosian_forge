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
def to_summary_by_func_name(cls, *, tasks: List[Dict]) -> 'TaskSummaries':
    summary = {}
    total_tasks = 0
    total_actor_tasks = 0
    total_actor_scheduled = 0
    for task in tasks:
        key = task['func_or_class_name']
        if key not in summary:
            summary[key] = TaskSummaryPerFuncOrClassName(func_or_class_name=task['func_or_class_name'], type=task['type'])
        task_summary = summary[key]
        state = task['state']
        if state not in task_summary.state_counts:
            task_summary.state_counts[state] = 0
        task_summary.state_counts[state] += 1
        type_enum = TaskType.DESCRIPTOR.values_by_name[task['type']].number
        if type_enum == TaskType.NORMAL_TASK:
            total_tasks += 1
        elif type_enum == TaskType.ACTOR_CREATION_TASK:
            total_actor_scheduled += 1
        elif type_enum == TaskType.ACTOR_TASK:
            total_actor_tasks += 1
    return TaskSummaries(summary=summary, total_tasks=total_tasks, total_actor_tasks=total_actor_tasks, total_actor_scheduled=total_actor_scheduled, summary_by='func_name')