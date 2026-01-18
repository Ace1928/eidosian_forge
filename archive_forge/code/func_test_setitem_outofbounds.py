import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_setitem_outofbounds():
    vec = robjects.r.seq(1, 10)
    with pytest.raises(IndexError):
        vec[20] = 20