from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
@staticmethod
def _bytes_to_int(data: bytes, mask: int) -> int:
    return int.from_bytes(data, 'big') & mask