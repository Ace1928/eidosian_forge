import pytest
import rpy2.robjects as robjects
import array
def test_rownames():
    m = robjects.r.matrix(robjects.IntVector(range(4)), nrow=2, ncol=2)
    assert m.rownames == rinterface.NULL
    m.rownames = robjects.StrVector(('c', 'd'))
    assert len(m.rownames) == 2
    assert m.rownames[0] == 'c'
    assert m.rownames[1] == 'd'
    with pytest.raises(ValueError):
        m.rownames = robjects.StrVector(('a', 'b', 'c'))