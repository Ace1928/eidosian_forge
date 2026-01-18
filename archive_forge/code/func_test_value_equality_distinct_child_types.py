import pytest
import cirq
def test_value_equality_distinct_child_types():
    v = {DistinctC(1): 4, DistinctCa(1): 5, DistinctCb(1): 6}
    assert v[DistinctC(1)] == 4
    assert v[DistinctCa(1)] == 5
    assert v[DistinctCb(1)] == 6
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(DistinctC(1), DistinctC(1))
    eq.add_equality_group(DistinctCa(1), DistinctCa(1))
    eq.add_equality_group(DistinctCb(1), DistinctCb(1))
    eq.add_equality_group(DistinctC(2))
    eq.add_equality_group(DistinctD(1))