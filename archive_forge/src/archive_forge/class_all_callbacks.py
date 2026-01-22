import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
class all_callbacks(StreamBasedExpressionVisitor):

    def __init__(self):
        self.ans = []
        super(all_callbacks, self).__init__()

    def enterNode(self, node):
        self.ans.append('Enter %s' % name(node))

    def exitNode(self, node, data):
        self.ans.append('Exit %s' % name(node))

    def beforeChild(self, node, child):
        self.ans.append('Before %s (from %s)' % (name(child), name(node)))

    def acceptChildResult(self, node, data, child_result):
        self.ans.append('Accept into %s' % name(node))

    def afterChild(self, node, child):
        self.ans.append('After %s (from %s)' % (name(child), name(node)))

    def finalizeResult(self, result):
        self.ans.append('Finalize')