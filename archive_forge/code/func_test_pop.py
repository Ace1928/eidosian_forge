import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_pop(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
    assert len(tl) == 3
    elt = tl.pop()
    assert elt == 3
    assert len(tl) == 2
    assert tl.tags == ('a', 'b')
    assert tuple(tl) == (1, 2)
    elt = tl.pop(0)
    assert elt == 1
    assert len(tl) == 1
    assert tl.tags == ('b',)