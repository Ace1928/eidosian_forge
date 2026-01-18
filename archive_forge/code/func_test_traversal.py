import tempfile
import os
import pickle
import random
import collections
import itertools
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.symbol_map import SymbolMap
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import (
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.suffix import suffix
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.parameter import parameter, parameter_dict, parameter_list
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.objective import objective, objective_dict, objective_list
from pyomo.core.kernel.variable import IVariable, variable, variable_dict, variable_list
from pyomo.core.kernel.block import IBlock, block, block_dict, block_tuple, block_list
from pyomo.core.kernel.sos import sos
from pyomo.opt.results import Solution
def test_traversal(self):
    b = block()
    b.v = variable()
    b.c1 = constraint()
    b.c1.deactivate()
    b.c2 = constraint_list()
    b.c2.append(constraint_list())
    b.B = block_list()
    b.B.append(block_list())
    b.B[0].append(block())
    b.B[0][0].c = constraint()
    b.B[0][0].b = block()
    b.B[0].deactivate()
    b._activate_large_storage_mode()

    def descend(obj):
        self.assertTrue(obj._is_container)
        return True
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(b, active=None, descend=descend)], [None, 'v', 'c1', 'c2', 'c2[0]', 'B', 'B[0]', 'B[0][0]', 'B[0][0].c', 'B[0][0].b'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(b, descend=descend)], [None, 'v', 'c2', 'c2[0]', 'B'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(b, active=True, descend=descend)], [None, 'v', 'c2', 'c2[0]', 'B'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(b, active=None, ctype=IConstraint, descend=descend)], [None, 'c1', 'c2', 'c2[0]', 'B', 'B[0]', 'B[0][0]', 'B[0][0].c', 'B[0][0].b'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(b, ctype=IConstraint, descend=descend)], [None, 'c2', 'c2[0]', 'B'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(b, ctype=IConstraint, active=True, descend=descend)], [None, 'c2', 'c2[0]', 'B'])
    m = pmo.block()
    m.B = pmo.block_list()
    m.B.append(pmo.block())
    m.B[0].v = pmo.variable()
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(m, ctype=IVariable)], [None, 'B', 'B[0]', 'B[0].v'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(m.B, ctype=IVariable)], ['B', 'B[0]', 'B[0].v'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(m.B[0], ctype=IVariable)], ['B[0]', 'B[0].v'])
    self.assertEqual([obj.name for obj in pmo.preorder_traversal(m.B[0].v, ctype=IVariable)], ['B[0].v'])