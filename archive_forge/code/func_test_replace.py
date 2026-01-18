import array
import pytest
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib.embedded
import rpy2.rlike.container as rlc
from rpy2 import robjects
from .. import utils
def test_replace():
    vec = robjects.vectors.IntVector(range(1, 6))
    i = array.array('i', [1, 3])
    vec.rx[rlc.TaggedList((i,))] = 20
    assert vec[0] == 20
    assert vec[1] == 2
    assert vec[2] == 20
    assert vec[3] == 4
    vec = robjects.vectors.IntVector(range(1, 6))
    i = array.array('i', [1, 5])
    vec.rx[rlc.TaggedList((i,))] = 50
    assert vec[0] == 50
    assert vec[1] == 2
    assert vec[2] == 3
    assert vec[3] == 4
    assert vec[4] == 50
    vec = robjects.vectors.IntVector(range(1, 6))
    vec.rx[1] = 70
    assert tuple(vec[0:5]) == (70, 2, 3, 4, 5)
    vec = robjects.vectors.IntVector(range(1, 6))
    vec.rx[robjects.vectors.IntVector((1, 3))] = 70
    assert tuple(vec[0:5]) == (70, 2, 70, 4, 5)
    m = robjects.r('matrix(1:10, ncol=2)')
    m.rx[1, 1] = 9
    assert m[0] == 9
    m = robjects.r('matrix(1:10, ncol=2)')
    m.rx[2, robjects.vectors.IntVector((1, 2))] = 9
    assert m[1] == 9
    assert m[6] == 9