from io import StringIO
from typing import Sequence, Dict, Optional, Mapping, MutableMapping
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver import results
from pyomo.contrib.solver import solution
import pyomo.environ as pyo
from pyomo.core.base.var import Var
class TestTerminationCondition(unittest.TestCase):

    def test_member_list(self):
        member_list = results.TerminationCondition._member_names_
        expected_list = ['unknown', 'convergenceCriteriaSatisfied', 'maxTimeLimit', 'iterationLimit', 'objectiveLimit', 'minStepLength', 'unbounded', 'provenInfeasible', 'locallyInfeasible', 'infeasibleOrUnbounded', 'error', 'interrupted', 'licensingProblems']
        self.assertEqual(member_list.sort(), expected_list.sort())

    def test_codes(self):
        self.assertEqual(results.TerminationCondition.unknown.value, 42)
        self.assertEqual(results.TerminationCondition.convergenceCriteriaSatisfied.value, 0)
        self.assertEqual(results.TerminationCondition.maxTimeLimit.value, 1)
        self.assertEqual(results.TerminationCondition.iterationLimit.value, 2)
        self.assertEqual(results.TerminationCondition.objectiveLimit.value, 3)
        self.assertEqual(results.TerminationCondition.minStepLength.value, 4)
        self.assertEqual(results.TerminationCondition.unbounded.value, 5)
        self.assertEqual(results.TerminationCondition.provenInfeasible.value, 6)
        self.assertEqual(results.TerminationCondition.locallyInfeasible.value, 7)
        self.assertEqual(results.TerminationCondition.infeasibleOrUnbounded.value, 8)
        self.assertEqual(results.TerminationCondition.error.value, 9)
        self.assertEqual(results.TerminationCondition.interrupted.value, 10)
        self.assertEqual(results.TerminationCondition.licensingProblems.value, 11)