from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
class LogicalConstraintsOnDisjuncts(unittest.TestCase):

    def test_logical_constraints_transformed(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        y1 = m.Y[1].get_associated_binary()
        y2 = m.Y[2].get_associated_binary()
        cons = hull.get_transformed_constraints(m.d[1]._logical_to_disjunctive.transformed_constraints[1])
        dis_z1 = hull.get_disaggregated_var(m.d[1]._logical_to_disjunctive.auxiliary_vars[1], m.d[1])
        dis_y1 = hull.get_disaggregated_var(y1, m.d[1])
        self.assertEqual(len(cons), 1)
        c = cons[0]
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, dis_z1 + dis_y1 - m.d[1].binary_indicator_var)
        cons = hull.get_transformed_constraints(m.d[1]._logical_to_disjunctive.transformed_constraints[2])
        self.assertEqual(len(cons), 1)
        c = cons[0]
        assertExpressionsStructurallyEqual(self, c.expr, dis_z1 - (1 - m.d[1].binary_indicator_var) * 0 >= m.d[1].binary_indicator_var)
        y1d = hull.get_disaggregated_var(y1, m.d[4])
        y2d = hull.get_disaggregated_var(y2, m.d[4])
        z1d = hull.get_disaggregated_var(m.d[4]._logical_to_disjunctive.auxiliary_vars[1], m.d[4])
        z2d = hull.get_disaggregated_var(m.d[4]._logical_to_disjunctive.auxiliary_vars[2], m.d[4])
        z3d = hull.get_disaggregated_var(m.d[4]._logical_to_disjunctive.auxiliary_vars[3], m.d[4])
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[1])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, -m.d[4].binary_indicator_var + z1d + y1d - y2d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[2])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, m.d[4].binary_indicator_var - y1d - z1d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[3])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, y2d - z1d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[4])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, -m.d[4].binary_indicator_var + z2d + y2d - y1d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[5])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, y1d - z2d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[6])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, m.d[4].binary_indicator_var - y2d - z2d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[7])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, z3d - z1d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[8])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
        assertExpressionsStructurallyEqual(self, simplified, z3d - z2d)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[9])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        assertExpressionsStructurallyEqual(self, cons.expr, 1 - z3d - (2 - (z1d + z2d)) - (1 - m.d[4].binary_indicator_var) * -1 <= 0 * m.d[4].binary_indicator_var)
        cons = hull.get_transformed_constraints(m.d[4]._logical_to_disjunctive.transformed_constraints[10])
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        assertExpressionsStructurallyEqual(self, cons.expr, z3d - (1 - m.d[4].binary_indicator_var) * 0 >= m.d[4].binary_indicator_var)
        self.assertFalse(m.bwahaha.active)
        self.assertFalse(m.p.active)

    @unittest.skipIf(not ct.linear_solvers, 'No linear solver available')
    def test_solution_obeys_logical_constraints(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        ct.check_solution_obeys_logical_constraints(self, 'hull', m)

    @unittest.skipIf(not ct.linear_solvers, 'No linear solver available')
    def test_boolean_vars_on_disjunct(self):
        m = models.makeBooleanVarsOnDisjuncts()
        ct.check_solution_obeys_logical_constraints(self, 'hull', m)

    def test_pickle(self):
        ct.check_transformed_model_pickles(self, 'hull')

    @unittest.skipIf(not dill_available, 'Dill is not available')
    def test_dill_pickle(self):
        ct.check_transformed_model_pickles_with_dill(self, 'hull')