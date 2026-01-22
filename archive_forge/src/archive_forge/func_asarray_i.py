import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray_i(self, data: Union[IntsXd, Sequence[Any]], *, dtype: Optional[DTypes]='int32') -> IntsXd:
    return cast(IntsXd, self.asarray(data, dtype=dtype))