import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class MongoDsn(AnyUrl):
    allowed_schemes = {'mongodb'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {'port': '27017'}