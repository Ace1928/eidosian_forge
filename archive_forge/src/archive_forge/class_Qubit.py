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
class Qubit(QubitState, Ket):
    """A multi-qubit ket in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    Examples
    ========

    Create a qubit in a couple of different ways and look at their attributes:

        >>> from sympy.physics.quantum.qubit import Qubit
        >>> Qubit(0,0,0)
        |000>
        >>> q = Qubit('0101')
        >>> q
        |0101>

        >>> q.nqubits
        4
        >>> len(q)
        4
        >>> q.dimension
        4
        >>> q.qubit_values
        (0, 1, 0, 1)

    We can flip the value of an individual qubit:

        >>> q.flip(1)
        |0111>

    We can take the dagger of a Qubit to get a bra:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> Dagger(q)
        <0101|
        >>> type(Dagger(q))
        <class 'sympy.physics.quantum.qubit.QubitBra'>

    Inner products work as expected:

        >>> ip = Dagger(q)*q
        >>> ip
        <0101|0101>
        >>> ip.doit()
        1
    """

    @classmethod
    def dual_class(self):
        return QubitBra

    def _eval_innerproduct_QubitBra(self, bra, **hints):
        if self.label == bra.label:
            return S.One
        else:
            return S.Zero

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        """Represent this qubits in the computational basis (ZGate).
        """
        _format = options.get('format', 'sympy')
        n = 1
        definite_state = 0
        for it in reversed(self.qubit_values):
            definite_state += n * it
            n = n * 2
        result = [0] * 2 ** self.dimension
        result[int(definite_state)] = 1
        if _format == 'sympy':
            return Matrix(result)
        elif _format == 'numpy':
            import numpy as np
            return np.array(result, dtype='complex').transpose()
        elif _format == 'scipy.sparse':
            from scipy import sparse
            return sparse.csr_matrix(result, dtype='complex').transpose()

    def _eval_trace(self, bra, **kwargs):
        indices = kwargs.get('indices', [])
        sorted_idx = list(indices)
        if len(sorted_idx) == 0:
            sorted_idx = list(range(0, self.nqubits))
        sorted_idx.sort()
        new_mat = self * bra
        for i in range(len(sorted_idx) - 1, -1, -1):
            new_mat = self._reduced_density(new_mat, int(sorted_idx[i]))
        if len(sorted_idx) == self.nqubits:
            return new_mat[0]
        else:
            return matrix_to_density(new_mat)

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