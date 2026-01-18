import pytest
import rpy2.robjects as robjects
import array
def test_tcrossprod():
    m = robjects.r.matrix(robjects.IntVector(range(4)), nrow=2)
    mtcp = m.tcrossprod(m)
    for i, val in enumerate((4, 6, 6, 10)):
        assert mtcp[i] == val