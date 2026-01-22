import itertools
import logging
import math
from io import StringIO
from contextlib import nullcontext
from pyomo.common.collections import OrderedSet
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
import pyomo.core.expr as EXPR
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
def _skip_trivial(constraint_data):
    if skip_trivial_constraints:
        if constraint_data._linear_canonical_form:
            repn = constraint_data.canonical_form()
            if repn.variables is None or len(repn.variables) == 0:
                return True
        elif constraint_data.body.polynomial_degree() == 0:
            return True
    return False