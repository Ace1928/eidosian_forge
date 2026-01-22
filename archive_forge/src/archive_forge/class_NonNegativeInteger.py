import re
import math
from typing import Any, Optional, SupportsFloat, SupportsInt, Union, Type
from ..helpers import NUMERIC_INF_OR_NAN, INVALID_NUMERIC, collapse_white_spaces
from .atomic_types import AnyAtomicType
class NonNegativeInteger(Integer):
    name = 'nonNegativeInteger'
    lower_bound = 0
    higher_bound: Optional[int] = None