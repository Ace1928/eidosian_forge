import copy
import pickle
import math
import os
from collections import defaultdict
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.environ import (
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.expr_common import ExpressionType, clone_counter
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import expression_to_string, clone_expression
from pyomo.core.expr import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numvalue import NumericValue
class decompose_linear_term_wrapper(object):

    def __init__(self, pairs):
        self.pairs = pairs

    def __eq__(self, other):
        if self.pairs is None:
            if other.pairs is not None:
                return False
        else:
            if other.pairs is None:
                return False
            if len(self.pairs) != len(other.pairs):
                return False
            for ndx in range(len(self.pairs)):
                if value(self.pairs[ndx][0]) != value(other.pairs[ndx][0]):
                    return False
                if self.pairs[ndx][1] is not other.pairs[ndx][1]:
                    return False
        return True