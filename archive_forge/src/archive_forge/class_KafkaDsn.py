import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class KafkaDsn(AnyUrl):
    allowed_schemes = {'kafka'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {'domain': 'localhost', 'port': '9092'}