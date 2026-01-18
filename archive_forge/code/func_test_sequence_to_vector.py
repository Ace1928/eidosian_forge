import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_sequence_to_vector():
    res = robjects.sequence_to_vector((1, 2, 3))
    assert isinstance(res, robjects.IntVector)
    res = robjects.sequence_to_vector((1, 2, 3.0))
    assert isinstance(res, robjects.FloatVector)
    res = robjects.sequence_to_vector(('ab', 'cd', 'ef'))
    assert isinstance(res, robjects.StrVector)
    with pytest.raises(ValueError):
        robjects.sequence_to_vector(list())