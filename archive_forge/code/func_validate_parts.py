import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
@classmethod
def validate_parts(cls, parts: 'Parts', validate_port: bool=True) -> 'Parts':
    return super().validate_parts(parts, validate_port=False)