import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_r_matmul_operator():
    m = robjects.r.matrix(robjects.IntVector(range(1, 5)), nrow=2)
    m2 = m.ro @ m
    for i, val in enumerate((7.0, 10.0, 15.0, 22.0)):
        assert m2[i] == val