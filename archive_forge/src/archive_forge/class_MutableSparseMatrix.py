from collections.abc import Callable
from sympy.core.containers import Dict
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .utilities import _iszero
from .decompositions import (
from .solvers import (
class MutableSparseMatrix(SparseRepMatrix, MutableRepMatrix):

    @classmethod
    def _new(cls, *args, **kwargs):
        rows, cols, smat = cls._handle_creation_inputs(*args, **kwargs)
        rep = cls._smat_to_DomainMatrix(rows, cols, smat)
        return cls._fromrep(rep)