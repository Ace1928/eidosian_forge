import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
class testAddDecisionRuleConstraints(unittest.TestCase):
    """
    Test method for adding decision rule equality constraints
    to the working model. There should be as many decision
    rule equality constraints as there are second-stage
    variables, and each constraint should relate a second-stage
    variable to the uncertain parameters and corresponding
    decision rule variables.
    """

    def make_simple_test_model(self):
        """
        Make simple model for DR constraint testing.
        """
        m = ConcreteModel()
        m.p = Param(range(3), initialize=0, mutable=True)
        m.z = Var([0, 1], initialize=0)
        m.util = Block()
        m.util.first_stage_variables = []
        m.util.second_stage_variables = list(m.z.values())
        m.util.uncertain_params = list(m.p.values())
        return m

    @unittest.skipIf(not scipy_available, 'Scipy is not available.')
    def test_num_dr_eqns_added_correct(self):
        """
        Check that number of DR equality constraints added
        by constraint declaration routines matches the number
        of second-stage variables in the model.
        """
        model_data = ROSolveResults()
        model_data.working_model = m = self.make_simple_test_model()
        m.decision_rule_var_0 = Var([0], initialize=0)
        m.decision_rule_var_1 = Var([0], initialize=0)
        m.util.decision_rule_vars = [m.decision_rule_var_0, m.decision_rule_var_1]
        config = Bunch()
        config.decision_rule_order = 0
        add_decision_rule_constraints(model_data=model_data, config=config)
        self.assertEqual(len(m.util.decision_rule_eqns), len(m.util.second_stage_variables), msg='The number of decision rule constraints added to model should equalthe number of control variables in the model.')

    @unittest.skipIf(not scipy_available, 'Scipy is not available.')
    def test_dr_eqns_form_correct(self):
        """
        Check that form of decision rule equality constraints
        is as expected.

        Decision rule equations should be of the standard form:
            (sum of DR monomial terms) - (second-stage variable) == 0
        where each monomial term should be of form:
            (product of uncertain parameters) * (decision rule variable)

        This test checks that the equality constraints are of this
        standard form.
        """
        model_data = ROSolveResults()
        model_data.working_model = m = self.make_simple_test_model()
        config = Bunch()
        config.decision_rule_order = 2
        add_decision_rule_variables(model_data, config)
        add_decision_rule_constraints(model_data, config)
        dr_monomial_param_combos = [(1,), (m.p[0],), (m.p[1],), (m.p[2],), (m.p[0], m.p[0]), (m.p[0], m.p[1]), (m.p[0], m.p[2]), (m.p[1], m.p[1]), (m.p[1], m.p[2]), (m.p[2], m.p[2])]
        dr_zip = zip(m.util.second_stage_variables, m.util.decision_rule_vars, m.util.decision_rule_eqns)
        for ss_var, indexed_dr_var, dr_eq in dr_zip:
            dr_eq_terms = dr_eq.body.args
            self.assertTrue(isinstance(dr_eq.body, SumExpression), msg=f'Body of DR constraint {dr_eq.name!r} is not of type {SumExpression.__name__}.')
            self.assertEqual(len(dr_eq_terms), len(dr_monomial_param_combos) + 1, msg=f'Number of additive terms in the DR expression of DR constraint with name {dr_eq.name!r} does not match expected value.')
            second_stage_var_term = dr_eq_terms[-1]
            last_term_is_neg_ss_var = isinstance(second_stage_var_term, MonomialTermExpression) and second_stage_var_term.args[0] == -1 and (second_stage_var_term.args[1] is ss_var) and (len(second_stage_var_term.args) == 2)
            self.assertTrue(last_term_is_neg_ss_var, msg=f'Last argument of last term in second-stage variableterm of DR constraint with name {dr_eq.name!r} is not the negative corresponding second-stage variable {ss_var.name!r}')
            dr_polynomial_terms = dr_eq_terms[:-1]
            dr_polynomial_zip = zip(dr_polynomial_terms, indexed_dr_var.values(), dr_monomial_param_combos)
            for idx, (term, dr_var, param_combo) in enumerate(dr_polynomial_zip):
                self.assertEqual(len(term.args), 2, msg=f'Length of `args` attribute of term {str(term)} of DR equation {dr_eq.name!r} is not as expected. Args: {term.args}')
                param_product_multiplicand = term.args[0]
                if idx == 0:
                    param_combo_found_in_term = (param_product_multiplicand,)
                    param_names = (str(param) for param in param_combo)
                elif len(param_combo) == 1:
                    param_combo_found_in_term = (param_product_multiplicand,)
                    param_names = (param.name for param in param_combo)
                else:
                    param_combo_found_in_term = param_product_multiplicand.args
                    param_names = (param.name for param in param_combo)
                self.assertEqual(param_combo_found_in_term, param_combo, msg=f'All but last multiplicand of DR monomial {str(term)} is not the uncertain parameter tuple ({', '.join(param_names)}).')
                dr_var_multiplicand = term.args[1]
                self.assertIs(dr_var_multiplicand, dr_var, msg=f'Last multiplicand of DR monomial {str(term)} is not the DR variable {dr_var.name!r}.')