import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def alloc1f(self, d0: int, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> Floats1d:
    return cast(Floats1d, self.alloc((d0,), dtype=dtype, zeros=zeros))