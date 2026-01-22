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
def _visit_vertex(self, u, leaf_to_root):
    if u in self._children:
        for v in self._children[u]:
            if v not in leaf_to_root:
                self._visit_vertex(v, leaf_to_root)
    leaf_to_root.add(u)