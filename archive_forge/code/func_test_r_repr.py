import array
import pytest
import rpy2.robjects as robjects
def test_r_repr():
    obj = robjects.baseenv['pi']
    s = obj.r_repr()
    assert s.startswith('3.14')