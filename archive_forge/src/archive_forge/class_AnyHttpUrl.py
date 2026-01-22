import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class AnyHttpUrl(AnyUrl):
    allowed_schemes = {'http', 'https'}
    __slots__ = ()