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
@validator('datatype', pre=True)
def uppercase_and_check_dtype(cls, v: str) -> str:
    if v.upper() not in REDIS_VECTOR_DTYPE_MAP:
        raise ValueError(f'datatype must be one of {REDIS_VECTOR_DTYPE_MAP.keys()}. Got {v}')
    return v.upper()