import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def alloc4i(self, d0: int, d1: int, d2: int, d3: int, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> Ints4d:
    return cast(Ints4d, self.alloc((d0, d1, d2, d3), dtype=dtype, zeros=zeros))