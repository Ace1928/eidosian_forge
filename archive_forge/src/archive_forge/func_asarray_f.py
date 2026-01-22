import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray_f(self, data: Union[FloatsXd, Sequence[Any]], *, dtype: Optional[DTypes]='float32') -> FloatsXd:
    return cast(FloatsXd, self.asarray(data, dtype=dtype))