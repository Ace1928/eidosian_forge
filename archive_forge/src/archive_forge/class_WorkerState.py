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
@dataclass(init=not IS_PYDANTIC_2)
class WorkerState(StateSchema):
    """Worker State"""
    worker_id: str = state_column(filterable=True)
    is_alive: bool = state_column(filterable=True)
    worker_type: TypeWorkerType = state_column(filterable=True)
    exit_type: Optional[TypeWorkerExitType] = state_column(filterable=True)
    node_id: str = state_column(filterable=True)
    ip: str = state_column(filterable=True)
    pid: int = state_column(filterable=True)
    exit_detail: Optional[str] = state_column(detail=True, filterable=False)
    worker_launch_time_ms: Optional[int] = state_column(filterable=False, detail=True, format_fn=Humanify.timestamp)
    worker_launched_time_ms: Optional[int] = state_column(filterable=False, detail=True, format_fn=Humanify.timestamp)
    start_time_ms: Optional[int] = state_column(filterable=False, detail=True, format_fn=Humanify.timestamp)
    end_time_ms: Optional[int] = state_column(filterable=False, detail=True, format_fn=Humanify.timestamp)
    debugger_port: Optional[int] = state_column(filterable=True, detail=True)
    num_paused_threads: Optional[int] = state_column(filterable=True, detail=True)