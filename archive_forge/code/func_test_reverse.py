import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_reverse(self):
    tn = ['a', 'b', 'c']
    tv = [1, 2, 3]
    tl = rlc.TaggedList(tv, tags=tn)
    tl.reverse()
    assert len(tl) == 3
    assert tl.tags == ('c', 'b', 'a')
    assert tuple(tl) == (3, 2, 1)