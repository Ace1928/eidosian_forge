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
class IndexedDisjunctions(unittest.TestCase):

    def setUp(self):
        random.seed(666)

    def test_disjunction_data_target(self):
        ct.check_disjunction_data_target(self, 'bigm')

    def test_disjunction_data_target_any_index(self):
        ct.check_disjunction_data_target_any_index(self, 'bigm')

    def check_trans_block_disjunctions_of_disjunct_datas(self, m):
        transBlock1 = m.component('_pyomo_gdp_bigm_reformulation')
        self.assertIsInstance(transBlock1, Block)
        self.assertIsInstance(transBlock1.component('relaxedDisjuncts'), Block)
        bigm = TransformationFactory('gdp.bigm')
        self.assertEqual(len(transBlock1.relaxedDisjuncts), 4)
        firstTerm1 = bigm.get_transformed_constraints(m.firstTerm[1].cons)
        self.assertIs(firstTerm1[0].parent_block(), transBlock1.relaxedDisjuncts[0])
        self.assertEqual(len(firstTerm1), 2)
        secondTerm1 = bigm.get_transformed_constraints(m.secondTerm[1].cons)
        self.assertIs(secondTerm1[0].parent_block(), transBlock1.relaxedDisjuncts[1])
        self.assertEqual(len(secondTerm1), 1)
        firstTerm2 = bigm.get_transformed_constraints(m.firstTerm[2].cons)
        self.assertIs(firstTerm2[0].parent_block(), transBlock1.relaxedDisjuncts[2])
        self.assertEqual(len(firstTerm2), 2)
        secondTerm2 = bigm.get_transformed_constraints(m.secondTerm[2].cons)
        self.assertIs(secondTerm2[0].parent_block(), transBlock1.relaxedDisjuncts[3])
        self.assertEqual(len(secondTerm2), 1)

    def test_simple_disjunction_of_disjunct_datas(self):
        ct.check_simple_disjunction_of_disjunct_datas(self, 'bigm')

    def test_any_indexed_disjunction_of_disjunct_datas(self):
        m = models.makeAnyIndexedDisjunctionOfDisjunctDatas()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        transBlock = m.component('_pyomo_gdp_bigm_reformulation')
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component('relaxedDisjuncts'), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 4)
        firstTerm1 = bigm.get_transformed_constraints(m.firstTerm[1].cons)
        self.assertIs(firstTerm1[0].parent_block(), transBlock.relaxedDisjuncts[0])
        self.assertEqual(len(firstTerm1), 2)
        secondTerm1 = bigm.get_transformed_constraints(m.secondTerm[1].cons)
        self.assertIs(secondTerm1[0].parent_block(), transBlock.relaxedDisjuncts[1])
        self.assertEqual(len(secondTerm1), 1)
        firstTerm2 = bigm.get_transformed_constraints(m.firstTerm[2].cons)
        self.assertIs(firstTerm2[0].parent_block(), transBlock.relaxedDisjuncts[2])
        self.assertEqual(len(firstTerm1), 2)
        secondTerm2 = bigm.get_transformed_constraints(m.secondTerm[2].cons)
        self.assertIs(secondTerm2[0].parent_block(), transBlock.relaxedDisjuncts[3])
        self.assertEqual(len(secondTerm1), 1)
        self.assertIsInstance(m.disjunction[1].algebraic_constraint.parent_component(), Constraint)
        self.assertIsInstance(m.disjunction[2].algebraic_constraint.parent_component(), Constraint)

    def check_first_iteration(self, model):
        transBlock = model.component('_pyomo_gdp_bigm_reformulation')
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component('disjunctionList_xor'), Constraint)
        self.assertEqual(len(transBlock.disjunctionList_xor), 1)
        self.assertFalse(model.disjunctionList[0].active)

    def check_second_iteration(self, model):
        transBlock = model.component('_pyomo_gdp_bigm_reformulation_4')
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component('relaxedDisjuncts'), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
        bigm = TransformationFactory('gdp.bigm')
        if model.component('firstTerm') is None:
            firstTerm1 = model.component('firstTerm[1]')
            secondTerm1 = model.component('secondTerm[1]')
        else:
            firstTerm1 = model.firstTerm[1]
            secondTerm1 = model.secondTerm[1]
        firstTerm = bigm.get_transformed_constraints(firstTerm1.cons)
        self.assertIs(firstTerm[0].parent_block(), transBlock.relaxedDisjuncts[0])
        self.assertEqual(len(firstTerm), 2)
        secondTerm = bigm.get_transformed_constraints(secondTerm1.cons)
        self.assertIs(secondTerm[0].parent_block(), transBlock.relaxedDisjuncts[1])
        self.assertEqual(len(secondTerm), 1)
        self.assertIsInstance(model.disjunctionList[1].algebraic_constraint.parent_component(), Constraint)
        self.assertIsInstance(model.disjunctionList[0].algebraic_constraint.parent_component(), Constraint)
        self.assertFalse(model.disjunctionList[1].active)
        self.assertFalse(model.disjunctionList[0].active)

    def test_disjunction_and_disjuncts_indexed_by_any(self):
        ct.check_disjunction_and_disjuncts_indexed_by_any(self, 'bigm')

    def test_iteratively_adding_disjunctions_transform_container(self):
        ct.check_iteratively_adding_disjunctions_transform_container(self, 'bigm')

    def test_iteratively_adding_disjunctions_transform_model(self):
        ct.check_iteratively_adding_disjunctions_transform_model(self, 'bigm')

    def test_iteratively_adding_to_indexed_disjunction_on_block(self):
        ct.check_iteratively_adding_to_indexed_disjunction_on_block(self, 'bigm')