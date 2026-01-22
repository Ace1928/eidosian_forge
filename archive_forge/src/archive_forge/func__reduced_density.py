import math
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import log
from sympy.core.basic import _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import Matrix, zeros
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.state import Ket, Bra, State
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import (
from mpmath.libmp.libintmath import bitcount
def _reduced_density(self, matrix, qubit, **options):
    """Compute the reduced density matrix by tracing out one qubit.
           The qubit argument should be of type Python int, since it is used
           in bit operations
        """

    def find_index_that_is_projected(j, k, qubit):
        bit_mask = 2 ** qubit - 1
        return (j >> qubit << 1 + qubit) + (j & bit_mask) + (k << qubit)
    old_matrix = represent(matrix, **options)
    old_size = old_matrix.cols
    new_size = old_size // 2
    new_matrix = Matrix().zeros(new_size)
    for i in range(new_size):
        for j in range(new_size):
            for k in range(2):
                col = find_index_that_is_projected(j, k, qubit)
                row = find_index_that_is_projected(i, k, qubit)
                new_matrix[i, j] += old_matrix[row, col]
    return new_matrix