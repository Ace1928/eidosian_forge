import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_remove(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
    assert len(tl) == 3
    tl.remove(2)
    assert len(tl) == 2
    assert tl.tags == ('a', 'c')
    assert tuple(tl) == (1, 3)