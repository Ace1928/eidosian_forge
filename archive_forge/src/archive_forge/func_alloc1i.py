import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def alloc1i(self, d0: int, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> Ints1d:
    return cast(Ints1d, self.alloc((d0,), dtype=dtype, zeros=zeros))