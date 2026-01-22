import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
class NLWriterInfo(object):
    """Return type for NLWriter.write()

    Attributes
    ----------
    variables: List[_VarData]

        The list of (unfixed) Pyomo model variables in the order written
        to the NL file

    constraints: List[_ConstraintData]

        The list of (active) Pyomo model constraints in the order written
        to the NL file

    objectives: List[_ObjectiveData]

        The list of (active) Pyomo model objectives in the order written
        to the NL file

    external_function_libraries: List[str]

        The list of paths to external function libraries referenced by
        the constraints / objectives written to the NL file

    row_labels: List[str]

        The list of string names for the constraints / objectives
        written to the NL file in the same order as
        :py:attr:`constraints` + :py:attr:`objectives` and the generated
        .row file.

    column_labels: List[str]

        The list of string names for the variables written to the NL
        file in the same order as the :py:attr:`variables` and generated
        .col file.

    eliminated_vars: List[Tuple[_VarData, NumericExpression]]

        The list of variables in the model that were eliminated by the
        presolve.  Each entry is a 2-tuple of (:py:class:`_VarData`,
        :py:class`NumericExpression`|`float`).  The list is in the
        necessary order for correct evaluation (i.e., all variables
        appearing in the expression must either have been sent to the
        solver, or appear *earlier* in this list.

    scaling: ScalingFactors or None

        namedtuple holding 3 lists of (variables, constraints, objectives)
        scaling factors in the same order (and size) as the `variables`,
        `constraints`, and `objectives` attributes above.

    """

    def __init__(self, var, con, obj, external_libs, row_labels, col_labels, eliminated_vars, scaling):
        self.variables = var
        self.constraints = con
        self.objectives = obj
        self.external_function_libraries = external_libs
        self.row_labels = row_labels
        self.column_labels = col_labels
        self.eliminated_vars = eliminated_vars
        self.scaling = scaling