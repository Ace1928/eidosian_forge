from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import polygamma
from sympy.functions.special.error_functions import (Si, Ci)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, i, j, a, b, c, d
from sympy.core import Pow
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
from sympy.tensor.array import Array
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter, _numpy_known_constants, \
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.testing.pytest import skip, raises
from sympy.external import import_module
def test_numpy_known_funcs_consts():
    assert _numpy_known_constants['NaN'] == 'numpy.nan'
    assert _numpy_known_constants['EulerGamma'] == 'numpy.euler_gamma'
    assert _numpy_known_functions['acos'] == 'numpy.arccos'
    assert _numpy_known_functions['log'] == 'numpy.log'