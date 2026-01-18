from sympy.core.numbers import (I, Rational, pi)
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.matrixutils import (
Get a cached matrix by name and format.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        format : str
            The format desired ('sympy', 'numpy', 'scipy.sparse')
        