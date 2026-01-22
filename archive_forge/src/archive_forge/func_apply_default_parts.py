import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
@classmethod
def apply_default_parts(cls, parts: 'Parts') -> 'Parts':
    for key, value in cls.get_default_parts(parts).items():
        if not parts[key]:
            parts[key] = value
    return parts