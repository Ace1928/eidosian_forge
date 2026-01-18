from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_write_exceptions(self):
    m = Block()
    with self.assertRaisesRegex(ValueError, '.*Could not infer file format from file name'):
        m.write(filename='foo.bogus')
    with self.assertRaisesRegex(ValueError, '.*Cannot write model in format'):
        m.write(format='bogus')