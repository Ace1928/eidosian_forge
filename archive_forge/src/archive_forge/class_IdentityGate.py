from itertools import chain
import random
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.power import Pow
from sympy.core.numbers import Number
from sympy.core.singleton import S as _S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import (UnitaryOperator, Operator,
from sympy.physics.quantum.matrixutils import matrix_tensor_product, matrix_eye
from sympy.physics.quantum.matrixcache import matrix_cache
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.iterables import is_sequence
class IdentityGate(OneQubitGate):
    """The single qubit identity gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = '1'
    gate_name_latex = '1'

    def _apply_operator_Qubit(self, qubits, **options):
        if qubits.nqubits < self.min_qubits:
            raise QuantumError('Gate needs a minimum of %r qubits to act on, got: %r' % (self.min_qubits, qubits.nqubits))
        return qubits

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('eye2', format)

    def _eval_commutator(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator(self, other, **hints):
        return Integer(2) * other