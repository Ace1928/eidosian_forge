import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_sample_probabilities_error_len():
    vec = robjects.IntVector(range(100))
    with pytest.raises(ValueError):
        vec.sample(10, probabilities=robjects.FloatVector([0.01] * 10))