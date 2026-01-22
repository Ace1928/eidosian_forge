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
class GetLogOptions:
    timeout: int
    node_id: Optional[str] = None
    node_ip: Optional[str] = None
    media_type: str = 'file'
    filename: Optional[str] = None
    actor_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt_number: int = 0
    pid: Optional[int] = None
    lines: int = 1000
    interval: Optional[float] = None
    suffix: str = 'out'
    submission_id: Optional[str] = None

    def __post_init__(self):
        if self.pid:
            self.pid = int(self.pid)
        if self.interval:
            self.interval = float(self.interval)
        self.lines = int(self.lines)
        if self.media_type == 'file':
            assert self.interval is None
        if self.media_type not in ['file', 'stream']:
            raise ValueError(f'Invalid media type: {self.media_type}')
        if not (self.node_id or self.node_ip) and (not (self.actor_id or self.task_id)):
            raise ValueError('node_id or node_ip must be provided as constructor arguments when no actor or task_id is supplied as arguments.')
        if self.node_id and self.node_ip:
            raise ValueError(f'Both node_id and node_ip are given. Only one of them can be provided. Given node id: {self.node_id}, given node ip: {self.node_ip}')
        if not (self.actor_id or self.task_id or self.pid or self.filename or self.submission_id):
            raise ValueError('None of actor_id, task_id, pid, submission_id or filename is provided. At least one of them is required to fetch logs.')
        if self.suffix not in ['out', 'err']:
            raise ValueError(f"Invalid suffix: {self.suffix}. Must be one of 'out' or 'err'.")