import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
@dataclasses.dataclass
class _QuadraticProcessedElements(_ProcessedElements):
    """Auxiliary data class for QuadraticBase._quadratic_flatten_once_and_add_to()."""
    quadratic_terms: DefaultDict['QuadraticTermKey', float] = dataclasses.field(default_factory=lambda: collections.defaultdict(float))