import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.acquaintance.strategies.cubic as ccasc
def test_skip_and_wrap_around():
    assert ccasc.skip_and_wrap_around(range(3)) == (0, 2, 1)
    assert ccasc.skip_and_wrap_around(range(4)) == (0, 3, 1, 2)
    assert ccasc.skip_and_wrap_around('abcde') == tuple('aebdc')
    assert ccasc.skip_and_wrap_around('abcdef') == tuple('afbecd')