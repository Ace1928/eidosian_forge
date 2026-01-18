from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32

        Generate the overloads for "next_(some type)" functions.
    