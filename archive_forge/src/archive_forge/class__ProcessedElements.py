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
class _ProcessedElements:
    """Auxiliary data class for LinearBase._flatten_once_and_add_to()."""
    terms: DefaultDict['Variable', float] = dataclasses.field(default_factory=lambda: collections.defaultdict(float))
    offset: float = 0.0