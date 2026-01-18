import pytest
import rpy2.robjects as robjects
import array
def test_nrow_get():
    m = robjects.r.matrix(robjects.IntVector(range(6)), nrow=3, ncol=2)
    assert m.nrow == 3