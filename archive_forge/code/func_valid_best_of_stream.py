import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('stream')
def valid_best_of_stream(cls, field_value, values):
    parameters = values['parameters']
    if parameters is not None and parameters.best_of is not None and (parameters.best_of > 1) and field_value:
        raise ValueError('`best_of` != 1 is not supported when `stream` == True')
    return field_value