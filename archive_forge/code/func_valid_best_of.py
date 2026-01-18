import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('best_of')
def valid_best_of(cls, field_value, values):
    if field_value is not None:
        if field_value <= 0:
            raise ValueError('`best_of` must be strictly positive')
        if field_value > 1 and values['seed'] is not None:
            raise ValueError('`seed` must not be set when `best_of` is > 1')
        sampling = values['do_sample'] | (values['temperature'] is not None) | (values['top_k'] is not None) | (values['top_p'] is not None) | (values['typical_p'] is not None)
        if field_value > 1 and (not sampling):
            raise ValueError('you must use sampling when `best_of` is > 1')
    return field_value