import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
class INFO(object):

    def __init__(self, symbolic=False):
        if symbolic:
            self.template = nl_writer.text_nl_debug_template
        else:
            self.template = nl_writer.text_nl_template
        self.subexpression_cache = {}
        self.subexpression_order = []
        self.external_functions = {}
        self.var_map = {}
        self.used_named_expressions = set()
        self.symbolic_solver_labels = symbolic
        self.visitor = nl_writer.AMPLRepnVisitor(self.template, self.subexpression_cache, self.subexpression_order, self.external_functions, self.var_map, self.used_named_expressions, self.symbolic_solver_labels, True, None)

    def __enter__(self):
        assert nl_writer.AMPLRepn.ActiveVisitor is None
        nl_writer.AMPLRepn.ActiveVisitor = self.visitor
        return self

    def __exit__(self, exc_type, exc_value, tb):
        assert nl_writer.AMPLRepn.ActiveVisitor is self.visitor
        nl_writer.AMPLRepn.ActiveVisitor = None