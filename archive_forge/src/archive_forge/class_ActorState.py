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
class ActorState(StateSchema):
    """Actor State"""
    actor_id: str = state_column(filterable=True)
    class_name: str = state_column(filterable=True)
    state: TypeActorStatus = state_column(filterable=True)
    job_id: str = state_column(filterable=True)
    name: Optional[str] = state_column(filterable=True)
    node_id: Optional[str] = state_column(filterable=True)
    pid: Optional[int] = state_column(filterable=True)
    ray_namespace: Optional[str] = state_column(filterable=True)
    serialized_runtime_env: Optional[str] = state_column(filterable=False, detail=True)
    required_resources: Optional[dict] = state_column(filterable=False, detail=True)
    death_cause: Optional[dict] = state_column(filterable=False, detail=True)
    is_detached: Optional[bool] = state_column(filterable=False, detail=True)
    placement_group_id: Optional[str] = state_column(detail=True, filterable=True)
    repr_name: Optional[str] = state_column(detail=True, filterable=True)