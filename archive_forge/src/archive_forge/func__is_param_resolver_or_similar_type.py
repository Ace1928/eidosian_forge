import numbers
from typing import Any, cast, Dict, Iterator, Mapping, Optional, TYPE_CHECKING, Union
import numpy as np
import sympy
from sympy.core import numbers as sympy_numbers
from cirq._compat import proper_repr
from cirq._doc import document
def _is_param_resolver_or_similar_type(obj: Any):
    return obj is None or isinstance(obj, (ParamResolver, dict))