from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
from sympy.testing.pytest import slow
def test_replace_subcircuit():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    circuit = (z, y, x, x)
    remove = (z, y, x)
    assert replace_subcircuit(circuit, Mul(*remove)) == (x,)
    assert replace_subcircuit(circuit, remove + (x,)) == ()
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    assert replace_subcircuit(circuit, remove, pos=0) == (x,)
    assert replace_subcircuit(circuit, (x, x), pos=2) == (z, y)
    assert replace_subcircuit(circuit, (h,)) == circuit
    circuit = (x, y, x, y, z)
    remove = (x, y, z)
    assert replace_subcircuit(Mul(*circuit), Mul(*remove)) == (x, y)
    remove = (x, y, x, y)
    assert replace_subcircuit(circuit, remove) == (z,)
    circuit = (x, h, cgate_z, h, cnot)
    remove = (x, h, cgate_z)
    assert replace_subcircuit(circuit, Mul(*remove), pos=-1) == (h, cnot)
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    remove = (h, h)
    assert replace_subcircuit(circuit, remove) == circuit
    remove = (h, cgate_z, h, cnot)
    assert replace_subcircuit(circuit, remove) == (x,)
    replace = (h, x)
    actual = replace_subcircuit(circuit, remove, replace=replace)
    assert actual == (x, h, x)
    circuit = (x, y, h, x, y, z)
    remove = (x, y)
    replace = (cnot, cgate_z)
    actual = replace_subcircuit(circuit, remove, replace=Mul(*replace))
    assert actual == (cnot, cgate_z, h, x, y, z)
    actual = replace_subcircuit(circuit, remove, replace=replace, pos=1)
    assert actual == (x, y, h, cnot, cgate_z, z)