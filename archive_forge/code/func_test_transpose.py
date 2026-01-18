import pytest
import rpy2.robjects as robjects
import array
def test_transpose():
    m = robjects.r.matrix(robjects.IntVector(range(6)), nrow=3, ncol=2)
    mt = m.transpose()
    for i, val in enumerate((0, 1, 2, 3, 4, 5)):
        assert m[i] == val
    for i, val in enumerate((0, 3, 1, 4, 2, 5)):
        assert mt[i] == val