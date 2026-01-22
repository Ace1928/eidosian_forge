import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray1i(self, data: Union[Ints1d, Sequence[int]], *, dtype: Optional[DTypes]='int32') -> Ints1d:
    return cast(Ints1d, self.asarray(data, dtype=dtype))