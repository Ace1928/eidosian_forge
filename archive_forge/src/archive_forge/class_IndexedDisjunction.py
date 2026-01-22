from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
@unittest.skipUnless(gurobi_available, 'Gurobi is not available')
class IndexedDisjunction(unittest.TestCase):

    def test_two_term_indexed_disjunction(self):
        """
        This test checks that we don't do anything silly with transformation Blocks in
        the case that the Disjunction is indexed.
        """
        m = make_indexed_equality_model()
        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m)
        cons = mbm.get_transformed_constraints(m.d[1].disjuncts[0].constraint[1])
        self.assertEqual(len(cons), 2)
        assertExpressionsEqual(self, cons[0].expr, m.x[1] >= m.d[1].disjuncts[0].binary_indicator_var + 2.0 * m.d[1].disjuncts[1].binary_indicator_var)
        assertExpressionsEqual(self, cons[1].expr, m.x[1] <= m.d[1].disjuncts[0].binary_indicator_var + 2.0 * m.d[1].disjuncts[1].binary_indicator_var)
        cons_again = mbm.get_transformed_constraints(m.d[1].disjuncts[1].constraint[1])
        self.assertEqual(len(cons_again), 2)
        self.assertIs(cons_again[0], cons[0])
        self.assertIs(cons_again[1], cons[1])
        cons = mbm.get_transformed_constraints(m.d[2].disjuncts[0].constraint[1])
        self.assertEqual(len(cons), 2)
        assertExpressionsEqual(self, cons[0].expr, m.x[2] >= m.d[2].disjuncts[0].binary_indicator_var + 2.0 * m.d[2].disjuncts[1].binary_indicator_var)
        assertExpressionsEqual(self, cons[1].expr, m.x[2] <= m.d[2].disjuncts[0].binary_indicator_var + 2.0 * m.d[2].disjuncts[1].binary_indicator_var)
        cons_again = mbm.get_transformed_constraints(m.d[2].disjuncts[1].constraint[1])
        self.assertEqual(len(cons_again), 2)
        self.assertIs(cons_again[0], cons[0])
        self.assertIs(cons_again[1], cons[1])