import ctypes
import logging
import os
from pyomo.common.fileutils import Library
from pyomo.core import value, Expression
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.common.collections import ComponentMap
class McCormick(object):
    """
    This class takes the constructed expression from MCPP_Visitor and
    allows for MC methods to be performed on pyomo expressions.

    __repn__(self): returns a display of an MC expression in the form:
    F: [lower interval : upper interval ] [convex underestimator :
    concave overestimator ] [ (convex subgradient) : (concave subgradient]

    lower(self): returns a float of the lower interval bound that is valid
    across the entire domain

    upper(self): returns a float of the upper interval bound that is valid
    across the entire domain

    concave(self): returns a float of the concave overestimator at the
    current value() of each variable.

    convex(self): returns a float of the convex underestimator at the
    current value() of each variable.

    ##Note: In order to describe the concave and convex relaxations over
    the entire domain, it is necessary to use changePoint() to repeat the
    calculation at different points.

    subcc(self): returns a ComponentMap() that maps the pyomo variables
    to the subgradients of the McCormick concave overestimators at the
    current value() of each variable.

    subcv(self): returns a ComponentMap() that maps the pyomo variables
    to the subgradients of the McCormick convex underestimators at the
    current value() of each variable.

    def changePoint(self, var, point): updates the current value() on the
    pyomo side and the current point on the MC++ side.
    """

    def __init__(self, expression, improved_var_bounds=None):
        self.mc_expr = None
        self.mcpp = _MCPP_lib()
        self.pyomo_expr = expression
        self.visitor = MCPP_visitor(expression, improved_var_bounds)
        self.mc_expr = self.visitor.walk_expression()

    def __del__(self):
        if self.mc_expr is not None:
            self.mcpp.release(self.mc_expr)
            self.mc_expr = None

    def __repn__(self):
        repn = self.mcpp.toString(self.mc_expr)
        repn = repn.decode('utf-8')
        return repn

    def __str__(self):
        return self.__repn__()

    def lower(self):
        return self.mcpp.lower(self.mc_expr)

    def upper(self):
        return self.mcpp.upper(self.mc_expr)

    def concave(self):
        self.warn_if_var_missing_value()
        return self.mcpp.concave(self.mc_expr)

    def convex(self):
        self.warn_if_var_missing_value()
        return self.mcpp.convex(self.mc_expr)

    def subcc(self):
        self.warn_if_var_missing_value()
        ans = ComponentMap()
        for key in self.visitor.var_to_idx:
            i = self.visitor.var_to_idx[key]
            ans[key] = self.mcpp.subcc(self.mc_expr, i)
        return ans

    def subcv(self):
        self.warn_if_var_missing_value()
        ans = ComponentMap()
        for key in self.visitor.var_to_idx:
            i = self.visitor.var_to_idx[key]
            ans[key] = self.mcpp.subcv(self.mc_expr, i)
        return ans

    def changePoint(self, var, point):
        var.set_value(point)
        self.visitor = MCPP_visitor(self.visitor.expr)
        self.mcpp.release(self.mc_expr)
        self.mc_expr = self.visitor.walk_expression()

    def warn_if_var_missing_value(self):
        if self.visitor.missing_value_warnings:
            for message in self.visitor.missing_value_warnings:
                logger.warning(message)
            self.visitor.missing_value_warnings = []