from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _powell(x, params, backend=math):
    A, exp = (params[0], backend.exp)
    return (A * x[0] * x[1] - 1, exp(-x[0]) + exp(-x[1]) - (1 + A ** (-1)))