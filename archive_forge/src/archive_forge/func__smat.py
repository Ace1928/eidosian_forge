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
@property
def _smat(self):
    sympy_deprecation_warning('\n            The private _smat attribute of SparseMatrix is deprecated. Use the\n            .todok() method instead.\n            ', deprecated_since_version='1.9', active_deprecations_target='deprecated-private-matrix-attributes')
    return self.todok()