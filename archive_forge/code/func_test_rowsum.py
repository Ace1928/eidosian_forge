import numpy as np
import pytest
import cirq
def test_rowsum():
    t = cirq.CliffordTableau(num_qubits=2)
    t._rowsum(0, 1)
    assert t.destabilizers()[0] == cirq.DensePauliString('XX', coefficient=1)
    t._rowsum(1, 2)
    assert t.destabilizers()[1] == cirq.DensePauliString('ZX', coefficient=1)
    t._rowsum(2, 3)
    assert t.stabilizers()[0] == cirq.DensePauliString('ZZ', coefficient=1)
    t = cirq.CliffordTableau(num_qubits=2)
    _S(t, 0)
    _CNOT(t, 0, 1)
    t._rowsum(0, 3)
    assert t.destabilizers()[0] == cirq.DensePauliString('XY', coefficient=1)
    t._rowsum(3, 0)
    assert t.stabilizers()[1] == cirq.DensePauliString('YX', coefficient=1)