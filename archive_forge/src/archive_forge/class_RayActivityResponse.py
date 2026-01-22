import concurrent.futures
from datetime import datetime
import enum
import logging
import json
import os
from typing import Optional
import aiohttp.web
from ray.dashboard.consts import RAY_CLUSTER_ACTIVITY_HOOK
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray._private.storage import _load_class
from ray.core.generated import gcs_service_pb2, gcs_service_pb2_grpc
from ray.dashboard.modules.job.common import JobInfoStorageClient
from ray._private.pydantic_compat import BaseModel, Extra, Field, validator
class RayActivityResponse(BaseModel, extra=Extra.allow):
    """
    Pydantic model used to inform if a particular Ray component can be considered
    active, and metadata about observation.
    """
    is_active: RayActivityStatus = Field(..., description='Whether the corresponding Ray component is considered active or inactive, or if there was an error while collecting this observation.')
    reason: Optional[str] = Field(None, description='Reason if Ray component is considered active or errored.')
    timestamp: float = Field(..., description='Timestamp of when this observation about the Ray component was made. This is in the format of seconds since unix epoch.')
    last_activity_at: Optional[float] = Field(None, description='Timestamp when last actvity of this Ray component finished in format of seconds since unix epoch. This field does not need to be populated for Ray components where it is not meaningful.')

    @validator('reason', always=True)
    def reason_required(cls, v, values, **kwargs):
        if 'is_active' in values and values['is_active'] != RayActivityStatus.INACTIVE:
            if v is None:
                raise ValueError('Reason is required if is_active is "active" or "error"')
        return v