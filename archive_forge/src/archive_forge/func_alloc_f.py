import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def alloc_f(self, shape: Shape, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> FloatsXd:
    return cast(FloatsXd, self.alloc(shape, dtype=dtype, zeros=zeros))