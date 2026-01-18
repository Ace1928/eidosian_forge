import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('truncate')
def valid_truncate(cls, v):
    if v is not None and v <= 0:
        raise ValueError('`truncate` must be strictly positive')
    return v