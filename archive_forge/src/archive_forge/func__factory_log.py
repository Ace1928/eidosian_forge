from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _factory_log(small):

    def _inner_factory(conds):
        precip = conds[0]

        def pre_processor(x, p):
            return (np.log(np.asarray(x) + math.exp(small)), p)

        def post_processor(x, p):
            return (np.exp(x), p)

        def fun(x, p):
            f = [None] * 3
            f[0] = math.exp(x[0]) + math.exp(x[2]) - p[0] - p[2]
            f[1] = math.exp(x[1]) + math.exp(x[2]) - p[1] - p[2]
            if precip:
                f[2] = x[0] + x[1] - math.log(p[3])
            else:
                f[2] = x[2] - small
            return f

        def jac(x, p):
            jout = np.empty((3, 3))
            jout[0, 0] = math.exp(x[0])
            jout[0, 1] = 0
            jout[0, 2] = math.exp(x[2])
            jout[1, 0] = 0
            jout[1, 1] = math.exp(x[1])
            jout[1, 2] = math.exp(x[2])
            if precip:
                jout[2, 0] = 1
                jout[2, 1] = 1
                jout[2, 2] = 0
            else:
                jout[2, 0] = 0
                jout[2, 1] = 0
                jout[2, 2] = 1
            return jout
        return NeqSys(3, 3, fun, jac, pre_processors=[pre_processor], post_processors=[post_processor])
    return _inner_factory