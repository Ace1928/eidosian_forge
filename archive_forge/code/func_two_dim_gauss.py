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
def two_dim_gauss(args, x, y, backend=None):
    A, x0, y0, sx, sy = args
    xp, yp = (x - x0, y - y0)
    vx, vy = (2 * sx ** 2, 2 * sy ** 2)
    return A * backend.exp(-(xp ** 2 / vx + yp ** 2 / vy))