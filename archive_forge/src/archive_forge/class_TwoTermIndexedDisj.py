from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
class TwoTermIndexedDisj(unittest.TestCase, CommonTests):

    def setUp(self):
        random.seed(666)
        self.pairs = [((0, 1, 'A'), 0), ((1, 1, 'A'), 1), ((0, 1, 'B'), 2), ((1, 1, 'B'), 3), ((0, 2, 'A'), 4), ((1, 2, 'A'), 5), ((0, 2, 'B'), 6), ((1, 2, 'B'), 7)]

    def test_xor_constraints(self):
        ct.check_indexed_xor_constraints(self, 'bigm')

    def test_deactivated_constraints(self):
        ct.check_constraints_deactivated_indexedDisjunction(self, 'bigm')

    def test_transformed_block_structure(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m.component('_pyomo_gdp_bigm_reformulation')
        self.assertIsInstance(transBlock, Block)
        disjBlock = transBlock.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 8)
        for i, j in self.pairs:
            self.assertEqual(len(disjBlock[j].component_map(Constraint)), 1)

    def test_disjunct_and_constraint_maps(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        disjBlock = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts
        oldblock = m.component('disjunct')
        for src, dest in self.pairs:
            srcDisjunct = oldblock[src]
            transformedDisjunct = disjBlock[dest]
            self.assertIs(bigm.get_src_disjunct(transformedDisjunct), srcDisjunct)
            self.assertIs(transformedDisjunct, srcDisjunct.transformation_block)
            transformed = bigm.get_transformed_constraints(srcDisjunct.c)
            if src[0]:
                self.assertEqual(len(transformed), 2)
                self.assertIsInstance(transformed[0], _ConstraintData)
                self.assertIsInstance(transformed[1], _ConstraintData)
                self.assertIs(bigm.get_src_constraint(transformed[0]), srcDisjunct.c)
                self.assertIs(bigm.get_src_constraint(transformed[1]), srcDisjunct.c)
            else:
                self.assertEqual(len(transformed), 1)
                self.assertIsInstance(transformed[0], _ConstraintData)
                self.assertIs(bigm.get_src_constraint(transformed[0]), srcDisjunct.c)

    def test_deactivated_disjuncts(self):
        ct.check_deactivated_disjuncts(self, 'bigm')

    def test_deactivated_disjunction(self):
        ct.check_deactivated_disjunctions(self, 'bigm')

    def test_create_using(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        self.diff_apply_to_and_create_using(m)