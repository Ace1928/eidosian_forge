from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing_extensions import TYPE_CHECKING, Literal
from langchain_community.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP
class RedisVectorField(RedisField):
    """Base class for Redis vector fields."""
    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default='FLOAT32')
    distance_metric: RedisDistanceMetric = Field(default='COSINE')
    initial_cap: Optional[int] = None

    @validator('algorithm', 'datatype', 'distance_metric', pre=True, each_item=True)
    def uppercase_strings(cls, v: str) -> str:
        return v.upper()

    @validator('datatype', pre=True)
    def uppercase_and_check_dtype(cls, v: str) -> str:
        if v.upper() not in REDIS_VECTOR_DTYPE_MAP:
            raise ValueError(f'datatype must be one of {REDIS_VECTOR_DTYPE_MAP.keys()}. Got {v}')
        return v.upper()

    def _fields(self) -> Dict[str, Any]:
        field_data = {'TYPE': self.datatype, 'DIM': self.dims, 'DISTANCE_METRIC': self.distance_metric}
        if self.initial_cap is not None:
            field_data['INITIAL_CAP'] = self.initial_cap
        return field_data