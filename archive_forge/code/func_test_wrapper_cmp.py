import itertools
import random
import pytest
import networkx
import cirq
def test_wrapper_cmp():
    u0 = cirq.contrib.Unique(0)
    u1 = cirq.contrib.Unique(1)
    u0, u1 = (u1, u0) if u1 < u0 else (u0, u1)
    assert u0 == u0
    assert u0 != u1
    assert u0 < u1
    assert u1 > u0
    assert u0 <= u0
    assert u0 <= u1
    assert u0 >= u0
    assert u1 >= u0