import cirq
from cirq.ops import qubit_manager as cqi
import pytest
@pytest.mark.parametrize('_', range(2))
def test_simple_qubit_manager(_):
    qm = cirq.ops.SimpleQubitManager()
    assert qm.qalloc(1) == [cqi.CleanQubit(0)]
    assert qm.qalloc(2) == [cqi.CleanQubit(1), cqi.CleanQubit(2)]
    assert qm.qalloc(1, dim=3) == [cqi.CleanQubit(3, dim=3)]
    assert qm.qborrow(1) == [cqi.BorrowableQubit(0)]
    assert qm.qborrow(2) == [cqi.BorrowableQubit(1), cqi.BorrowableQubit(2)]
    assert qm.qborrow(1, dim=3) == [cqi.BorrowableQubit(3, dim=3)]
    qm.qfree([cqi.CleanQubit(i) for i in range(3)] + [cqi.CleanQubit(3, dim=3)])
    qm.qfree([cqi.BorrowableQubit(i) for i in range(3)] + [cqi.BorrowableQubit(3, dim=3)])
    with pytest.raises(ValueError, match='not allocated'):
        qm.qfree([cqi.CleanQubit(10)])
    with pytest.raises(ValueError, match='not allocated'):
        qm.qfree([cqi.BorrowableQubit(10)])