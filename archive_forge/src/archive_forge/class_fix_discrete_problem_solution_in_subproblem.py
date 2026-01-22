from contextlib import contextmanager
import logging
from math import fabs
import sys
from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
from pyomo.core.expr.numvalue import native_types
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.opt import SolverFactory
class fix_discrete_problem_solution_in_subproblem(fix_discrete_solution_in_subproblem):

    def __init__(self, discrete_prob_util_block, subproblem_util_block, solver, config):
        self.discrete_prob_util_block = discrete_prob_util_block
        self.subprob_util_block = subproblem_util_block
        self.solver = solver
        self.config = config

    def __enter__(self):
        fixed = []
        for disjunct, block in zip(self.discrete_prob_util_block.disjunct_list, self.subprob_util_block.disjunct_list):
            if not disjunct.indicator_var.value:
                block.deactivate()
                block.binary_indicator_var.fix(0)
            else:
                block.binary_indicator_var.fix(1)
                fixed.append(block.name)
        self.config.logger.debug("Fixed the following Disjuncts to 'True': %s" % ', '.join(fixed))
        fixed_bools = []
        for discrete_problem_bool, subprob_bool in zip(self.discrete_prob_util_block.non_indicator_boolean_variable_list, self.subprob_util_block.non_indicator_boolean_variable_list):
            discrete_problem_binary = discrete_problem_bool.get_associated_binary()
            subprob_binary = subprob_bool.get_associated_binary()
            val = discrete_problem_binary.value
            if val is None:
                discrete_problem_binary.set_value(1)
                subprob_binary.fix(1)
                bool_val = True
            elif val > 0.5:
                subprob_binary.fix(1)
                bool_val = True
            else:
                subprob_binary.fix(0)
                bool_val = False
            fixed_bools.append('%s = %s' % (subprob_bool.name, bool_val))
        self.config.logger.debug('Fixed the following Boolean variables: %s' % ', '.join(fixed_bools))
        if self.config.force_subproblem_nlp:
            fixed_discrete = []
            for discrete_problem_var, subprob_var in zip(self.discrete_prob_util_block.discrete_variable_list, self.subprob_util_block.discrete_variable_list):
                fix_discrete_var(subprob_var, discrete_problem_var.value, self.config)
                fixed_discrete.append('%s = %s' % (subprob_var.name, discrete_problem_var.value))
            self.config.logger.debug('Fixed the following integer variables: %s' % ', '.join(fixed_discrete))
        self.config.subproblem_initialization_method(self.solver, self.subprob_util_block, self.discrete_prob_util_block)
        return self