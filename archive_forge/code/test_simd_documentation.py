import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__

            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            