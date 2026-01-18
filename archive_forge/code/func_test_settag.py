import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_settag(self):
    tn = ['a', 'b', 'c']
    tv = [1, 2, 3]
    tl = rlc.TaggedList(tv, tags=tn)
    tl.settag(1, 'z')
    assert tl.tags == ('a', 'z', 'c')