import collections
import logging
from operator import attrgetter
from pyomo.common.config import (
from pyomo.common.dependencies import scipy, numpy as np
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
class LinearStandardFormInfo(object):
    """Return type for LinearStandardFormCompiler.write()

    Attributes
    ----------
    c : scipy.sparse.csr_array

        The objective coefficients.  Note that this is a sparse array
        and may contain multiple rows (for multiobjective problems).  The
        objectives may be calculated by "c @ x"

    A : scipy.sparse.csc_array

        The constraint coefficients.  The constraint bodies may be
        calculated by "A @ x"

    rhs : numpy.ndarray

        The constraint right-hand sides.

    rows : List[Tuple[_ConstraintData, int]]

        The list of Pyomo constraint objects corresponding to the rows
        in `A`.  Each element in the list is a 2-tuple of
        (_ConstraintData, row_multiplier).  The `row_multiplier` will be
        +/- 1 indicating if the row was multiplied by -1 (corresponding
        to a constraint lower bound) or +1 (upper bound).

    columns : List[_VarData]

        The list of Pyomo variable objects corresponding to columns in
        the `A` and `c` matrices.

    eliminated_vars: List[Tuple[_VarData, NumericExpression]]

        The list of variables from the original model that do not appear
        in the standard form (usually because they were replaced by
        nonnegative variables).  Each entry is a 2-tuple of
        (:py:class:`_VarData`, :py:class`NumericExpression`|`float`).
        The list is in the necessary order for correct evaluation (i.e.,
        all variables appearing in the expression must either have
        appeared in the standard form, or appear *earlier* in this list.

    """

    def __init__(self, c, A, rhs, rows, columns, eliminated_vars):
        self.c = c
        self.A = A
        self.rhs = rhs
        self.rows = rows
        self.columns = columns
        self.eliminated_vars = eliminated_vars

    @property
    def x(self):
        return self.columns

    @property
    def b(self):
        return self.rhs