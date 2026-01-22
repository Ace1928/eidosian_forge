import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class _MapUnit:

    def __init__(self, expr: TuningParameterExpression):
        self.expr = expr
        self.positions: List[List[Any]] = []

    def __eq__(self, other: Any) -> bool:
        return self.expr == other.expr and self.positions == other.positions

    def __uuid__(self) -> str:
        return to_uuid(self.expr, self.positions)

    def copy(self) -> '_MapUnit':
        res = _MapUnit(self.expr)
        res.positions = list(self.positions)
        return res