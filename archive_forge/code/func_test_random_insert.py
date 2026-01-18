from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
from sympy.testing.pytest import slow
@slow
def test_random_insert():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    choices = [(x, x)]
    circuit = (y, y)
    loc, choice = (0, 0)
    actual = random_insert(circuit, choices, seed=[loc, choice])
    assert actual == (x, x, y, y)
    circuit = (x, y, z, h)
    choices = [(h, h), (x, y, z)]
    expected = (x, x, y, z, y, z, h)
    loc, choice = (1, 1)
    actual = random_insert(circuit, choices, seed=[loc, choice])
    assert actual == expected
    gate_list = [x, y, z, h, cnot, cgate_z]
    ids = list(bfs_identity_search(gate_list, 2, max_depth=4))
    eq_ids = flatten_ids(ids)
    circuit = (x, y, h, cnot, cgate_z)
    expected = (x, z, x, z, x, y, h, cnot, cgate_z)
    loc, choice = (1, 30)
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    assert actual == expected
    circuit = Mul(*circuit)
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    assert actual == expected