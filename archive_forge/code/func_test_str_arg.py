import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
def test_str_arg():
    PolyT = create_Poly('x')
    p = PolyT([1, 2, 3])
    x = Log10('T')
    res = p({'x': x, 'T': 1000})
    ref = 1 + 2 * 3 + 3 * 9
    assert abs(res - ref) < 1e-12