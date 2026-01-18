import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_sort(self):
    tn = ['a', 'c', 'b']
    tv = [1, 3, 2]
    tl = rlc.TaggedList(tv, tags=tn)
    tl.sort()
    assert tl.tags == ('a', 'b', 'c')
    assert tuple(tl) == (1, 2, 3)