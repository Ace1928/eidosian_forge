import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_setnames():
    vec = robjects.vectors.IntVector(array.array('i', [1, 2, 3]))
    names = ['x', 'y', 'z']
    vec.names = names
    for i in range(len(vec)):
        assert names[i] == vec.names[i]