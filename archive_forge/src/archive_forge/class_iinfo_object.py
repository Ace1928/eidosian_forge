from __future__ import annotations
from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union
import cupy as np
@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int