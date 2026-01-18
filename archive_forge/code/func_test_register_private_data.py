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
def test_register_private_data(self):
    _save = Block._private_data_initializers
    Block._private_data_initializers = pdi = _save.copy()
    pdi.clear()
    try:
        self.assertEqual(len(pdi), 0)
        b = Block(concrete=True)
        ps = b.private_data()
        self.assertEqual(ps, {})
        self.assertEqual(len(pdi), 1)
    finally:
        Block._private_data_initializers = _save

    def init():
        return {'a': None, 'b': 1}
    Block._private_data_initializers = pdi = _save.copy()
    pdi.clear()
    try:
        self.assertEqual(len(pdi), 0)
        Block.register_private_data_initializer(init)
        self.assertEqual(len(pdi), 1)
        b = Block(concrete=True)
        ps = b.private_data()
        self.assertEqual(ps, {'a': None, 'b': 1})
        self.assertEqual(len(pdi), 1)
    finally:
        Block._private_data_initializers = _save
    Block._private_data_initializers = pdi = _save.copy()
    pdi.clear()
    try:
        Block.register_private_data_initializer(init)
        self.assertEqual(len(pdi), 1)
        Block.register_private_data_initializer(init, 'pyomo')
        self.assertEqual(len(pdi), 2)
        with self.assertRaisesRegex(RuntimeError, "Duplicate initializer registration for 'private_data' dictionary \\(scope=pyomo.core.tests.unit.test_block\\)"):
            Block.register_private_data_initializer(init)
        with self.assertRaisesRegex(ValueError, "'private_data' scope must be substrings of the caller's module name. Received 'invalid' when calling register_private_data_initializer\\(\\)."):
            Block.register_private_data_initializer(init, 'invalid')
        self.assertEqual(len(pdi), 2)
    finally:
        Block._private_data_initializers = _save