import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_r_and():
    v = robjects.vectors.BoolVector((True, False))
    res = v.ro & True
    assert all((x is y for x, y in zip(res, (True, False))))