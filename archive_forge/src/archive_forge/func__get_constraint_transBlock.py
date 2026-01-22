from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct
import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus
from weakref import ref as weakref_ref
from collections import defaultdict
import logging
def _get_constraint_transBlock(constraint):
    parent_disjunct = _find_parent_disjunct(constraint)
    transBlock = parent_disjunct._transformation_block
    if transBlock is None:
        raise GDP_Error("Constraint '%s' is on a disjunct which has not been transformed" % constraint.name)
    transBlock = transBlock()
    return transBlock