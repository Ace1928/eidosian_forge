import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray2f(self, data: Union[Floats2d, Sequence[Sequence[float]]], *, dtype: Optional[DTypes]='float32') -> Floats2d:
    return cast(Floats2d, self.asarray(data, dtype=dtype))