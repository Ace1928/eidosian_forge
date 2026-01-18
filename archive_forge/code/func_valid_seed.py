import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('seed')
def valid_seed(cls, v):
    if v is not None and v < 0:
        raise ValueError('`seed` must be positive')
    return v