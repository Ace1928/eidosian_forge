from __future__ import annotations
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import ModuleType
from typing import (
import numpy as np
@runtime_checkable
class _chunkedarrayfunction(_arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Chunked duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> _Chunks:
        ...