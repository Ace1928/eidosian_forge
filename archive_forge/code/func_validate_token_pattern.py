import inspect
import re
from collections import defaultdict
from enum import Enum
from typing import (
from thinc.api import ConfigValidationError, Model, Optimizer
from thinc.config import Promise
from .attrs import NAMES
from .compat import Literal
from .lookups import Lookups
from .util import is_cython_func
def validate_token_pattern(obj: list) -> List[str]:
    get_key = lambda k: NAMES[k] if isinstance(k, int) and k < len(NAMES) else k
    if isinstance(obj, list):
        converted = []
        for pattern in obj:
            if isinstance(pattern, dict):
                pattern = {get_key(k): v for k, v in pattern.items()}
            converted.append(pattern)
        obj = converted
    return validate(TokenPatternSchema, {'pattern': obj})