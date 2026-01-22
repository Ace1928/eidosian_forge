import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def _get_batch_sizes(self, length: int, sizes: Iterator[int]):
    output = []
    i = 0
    while i < length:
        output.append(next(sizes))
        i += output[-1]
    return output