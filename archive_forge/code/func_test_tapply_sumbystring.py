import pytest
import rpy2.rlike.functional as rlf
def test_tapply_sumbystring():
    seq = (1, 2, 3, 4, 5, 6)
    tags = ('a', 'b', 'a', 'c', 'b', 'a')
    expected = {'a': 1 + 3 + 6, 'b': 2 + 5, 'c': 4}
    res = rlf.tapply(seq, tags, sum)
    for k, v in res:
        assert expected[k] == v