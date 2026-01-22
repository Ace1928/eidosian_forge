import logging
import sys
import types
from math import fabs
from weakref import ref as weakref_ref
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.errors import PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import native_logical_types, native_types
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core import (
from pyomo.core.base.component import (
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.core.expr.expr_common import ExpressionType
class _DisjunctionData(ActiveComponentData):
    __slots__ = ('disjuncts', 'xor', '_algebraic_constraint', '_transformation_map')
    __autoslot_mappers__ = {'_algebraic_constraint': AutoSlots.weakref_mapper}
    _NoArgument = (0,)

    @property
    def algebraic_constraint(self):
        return None if self._algebraic_constraint is None else self._algebraic_constraint()

    def __init__(self, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._active = True
        self.disjuncts = []
        self.xor = True
        self._algebraic_constraint = None
        self._transformation_map = {}

    def set_value(self, expr):
        for e in expr:
            if hasattr(e, 'is_component_type') and e.is_component_type():
                if e.ctype == Disjunct and (not e.is_indexed()):
                    self.disjuncts.append(e)
                    continue
                e_iter = [e]
            elif hasattr(e, '__iter__'):
                e_iter = e
            else:
                e_iter = [e]
            expressions = []
            for _tmpe in e_iter:
                try:
                    if _tmpe.is_expression_type():
                        expressions.append(_tmpe)
                        continue
                except AttributeError:
                    pass
                msg = " in '%s'" % (type(e).__name__,) if e_iter is e else ''
                raise ValueError("Unexpected term for Disjunction '%s'.\n    Expected a Disjunct object, relational expression, or iterable of\n    relational expressions but got '%s'%s" % (self.name, type(_tmpe).__name__, msg))
            comp = self.parent_component()
            if comp._autodisjuncts is None:
                b = self.parent_block()
                comp._autodisjuncts = Disjunct(Any)
                b.add_component(unique_component_name(b, comp.local_name + '_disjuncts'), comp._autodisjuncts)
                comp._autodisjuncts.construct()
            disjunct = comp._autodisjuncts[len(comp._autodisjuncts)]
            disjunct.constraint = c = ConstraintList()
            disjunct.propositions = p = LogicalConstraintList()
            for e in expressions:
                if e.is_expression_type(ExpressionType.RELATIONAL):
                    c.add(e)
                elif e.is_expression_type(ExpressionType.LOGICAL):
                    p.add(e)
                else:
                    raise RuntimeError(f'Unsupported expression type on Disjunct {disjunct.name}: expected either relational or logical expression, found {e.__class__.__name__}')
            self.disjuncts.append(disjunct)