import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
Wraps around `cupy.RawModule` and `cupy.RawKernel` to verify CuPy availability
    and lazily compile the latter on first invocation.

    The default CuPy behaviour triggers the compilation as soon as the `cupy.RawKernel` object
    is accessed.