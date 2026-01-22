from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
class _ExternalFunctionVisitor(StreamBasedExpressionVisitor):

    def initializeWalker(self, expr):
        self._functions = []
        self._seen = set()
        return (True, None)

    def exitNode(self, node, data):
        if type(node) is ExternalFunctionExpression:
            if id(node) not in self._seen:
                self._seen.add(id(node))
                self._functions.append(node)

    def finalizeResult(self, result):
        return self._functions

    def enterNode(self, node):
        pass

    def acceptChildResult(self, node, data, child_result, child_idx):
        pass

    def acceptChildResult(self, node, data, child_result, child_idx):
        if child_result.__class__ in native_types:
            return (False, None)
        return (child_result.is_expression_type(), None)