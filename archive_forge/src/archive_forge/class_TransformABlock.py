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
class TransformABlock(unittest.TestCase, CommonTests):

    def test_transformation_simple_block(self):
        ct.check_transformation_simple_block(self, 'hull')

    def test_transform_block_data(self):
        ct.check_transform_block_data(self, 'hull')

    def test_simple_block_target(self):
        ct.check_simple_block_target(self, 'hull')

    def test_block_data_target(self):
        ct.check_block_data_target(self, 'hull')

    def test_indexed_block_target(self):
        ct.check_indexed_block_target(self, 'hull')

    def test_block_targets_inactive(self):
        ct.check_block_targets_inactive(self, 'hull')

    def test_block_only_targets_transformed(self):
        ct.check_block_only_targets_transformed(self, 'hull')

    def test_create_using(self):
        m = models.makeTwoTermDisjOnBlock()
        ct.diff_apply_to_and_create_using(self, m, 'gdp.hull')