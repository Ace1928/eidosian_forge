from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import (apply_grover, superposition_basis,
def test_superposition_basis():
    nbits = 2
    first_half_state = IntQubit(0, nqubits=nbits) / 2 + IntQubit(1, nqubits=nbits) / 2
    second_half_state = IntQubit(2, nbits) / 2 + IntQubit(3, nbits) / 2
    assert first_half_state + second_half_state == superposition_basis(nbits)
    nbits = 3
    firstq = 1 / sqrt(8) * IntQubit(0, nqubits=nbits) + 1 / sqrt(8) * IntQubit(1, nqubits=nbits)
    secondq = 1 / sqrt(8) * IntQubit(2, nbits) + 1 / sqrt(8) * IntQubit(3, nbits)
    thirdq = 1 / sqrt(8) * IntQubit(4, nbits) + 1 / sqrt(8) * IntQubit(5, nbits)
    fourthq = 1 / sqrt(8) * IntQubit(6, nbits) + 1 / sqrt(8) * IntQubit(7, nbits)
    assert firstq + secondq + thirdq + fourthq == superposition_basis(nbits)