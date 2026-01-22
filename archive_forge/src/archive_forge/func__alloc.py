import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def _alloc(shape, dtype, *, zeros: bool=True):
    if zeros:
        return cupy.zeros(shape, dtype)
    else:
        return cupy.empty(shape, dtype)