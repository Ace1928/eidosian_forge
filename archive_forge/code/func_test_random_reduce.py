from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
from sympy.testing.pytest import slow
@slow
def test_random_reduce():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    gate_list = [x, y, z]
    ids = list(bfs_identity_search(gate_list, 1, max_depth=4))
    circuit = (x, y, h, z, cnot)
    assert random_reduce(circuit, []) == circuit
    assert random_reduce(circuit, ids) == circuit
    seq = [2, 11, 9, 3, 5]
    circuit = (x, y, z, x, y, h)
    assert random_reduce(circuit, ids, seed=seq) == (x, y, h)
    circuit = (x, x, y, y, z, z)
    assert random_reduce(circuit, ids, seed=seq) == (x, x, y, y)
    seq = [14, 13, 0]
    assert random_reduce(circuit, ids, seed=seq) == (y, y, z, z)
    gate_list = [x, y, z, h, cnot, cgate_z]
    ids = list(bfs_identity_search(gate_list, 2, max_depth=4))
    seq = [25]
    circuit = (x, y, z, y, h, y, h, cgate_z, h, cnot)
    expected = (x, y, z, cgate_z, h, cnot)
    assert random_reduce(circuit, ids, seed=seq) == expected
    circuit = Mul(*circuit)
    assert random_reduce(circuit, ids, seed=seq) == expected