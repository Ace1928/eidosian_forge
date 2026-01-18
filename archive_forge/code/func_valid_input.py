import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('inputs')
def valid_input(cls, v):
    if not v:
        raise ValueError('`inputs` cannot be empty')
    return v