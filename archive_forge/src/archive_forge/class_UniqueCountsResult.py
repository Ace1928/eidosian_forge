from __future__ import annotations
from ._array_object import Array
from typing import NamedTuple
import cupy as np
class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array