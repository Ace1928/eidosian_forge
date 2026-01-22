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
class TestHeterogeneousContainer(unittest.TestCase):
    model = pmo.block()
    model.v = pmo.variable()
    model.V = pmo.variable_list()
    model.V.append(pmo.variable())
    model.V.append(pmo.variable_list())
    model.V[1].append(pmo.variable())
    model.c = pmo.constraint()
    model.C = pmo.constraint_list()
    model.C.append(pmo.constraint())
    model.C.append(pmo.constraint_list())
    model.C[1].append(pmo.constraint())
    b_clone = model.clone()
    model.b = b_clone.clone()
    model.B = pmo.block_list()
    model.B.append(b_clone.clone())
    model.B.append(pmo.block_list())
    model.B[1].append(b_clone.clone())
    del b_clone
    model.j = junk()
    model.J = junk_list()
    model.J.append(junk())
    model.J.append(junk_list())
    model.J[1].append(junk())
    model.J[1][0].b = pmo.block()
    model.J[1][0].b.v = pmo.variable()
    model_clone = model.clone()
    model.k = pmo.block()
    model.K = pmo.block_list()
    model.K.append(model_clone.clone())
    del model_clone

    def test_preorder_traversal(self):
        model = self.model.clone()
        order = list((str(obj) for obj in pmo.preorder_traversal(model)))
        self.assertEqual(order, ['<block>', 'v', 'V', 'V[0]', 'V[1]', 'V[1][0]', 'c', 'C', 'C[0]', 'C[1]', 'C[1][0]', 'b', 'b.v', 'b.V', 'b.V[0]', 'b.V[1]', 'b.V[1][0]', 'b.c', 'b.C', 'b.C[0]', 'b.C[1]', 'b.C[1][0]', 'B', 'B[0]', 'B[0].v', 'B[0].V', 'B[0].V[0]', 'B[0].V[1]', 'B[0].V[1][0]', 'B[0].c', 'B[0].C', 'B[0].C[0]', 'B[0].C[1]', 'B[0].C[1][0]', 'B[1]', 'B[1][0]', 'B[1][0].v', 'B[1][0].V', 'B[1][0].V[0]', 'B[1][0].V[1]', 'B[1][0].V[1][0]', 'B[1][0].c', 'B[1][0].C', 'B[1][0].C[0]', 'B[1][0].C[1]', 'B[1][0].C[1][0]', 'j', 'J', 'J[0]', 'J[1]', 'J[1][0]', 'J[1][0].b', 'J[1][0].b.v', 'k', 'K', 'K[0]', 'K[0].v', 'K[0].V', 'K[0].V[0]', 'K[0].V[1]', 'K[0].V[1][0]', 'K[0].c', 'K[0].C', 'K[0].C[0]', 'K[0].C[1]', 'K[0].C[1][0]', 'K[0].b', 'K[0].b.v', 'K[0].b.V', 'K[0].b.V[0]', 'K[0].b.V[1]', 'K[0].b.V[1][0]', 'K[0].b.c', 'K[0].b.C', 'K[0].b.C[0]', 'K[0].b.C[1]', 'K[0].b.C[1][0]', 'K[0].B', 'K[0].B[0]', 'K[0].B[0].v', 'K[0].B[0].V', 'K[0].B[0].V[0]', 'K[0].B[0].V[1]', 'K[0].B[0].V[1][0]', 'K[0].B[0].c', 'K[0].B[0].C', 'K[0].B[0].C[0]', 'K[0].B[0].C[1]', 'K[0].B[0].C[1][0]', 'K[0].B[1]', 'K[0].B[1][0]', 'K[0].B[1][0].v', 'K[0].B[1][0].V', 'K[0].B[1][0].V[0]', 'K[0].B[1][0].V[1]', 'K[0].B[1][0].V[1][0]', 'K[0].B[1][0].c', 'K[0].B[1][0].C', 'K[0].B[1][0].C[0]', 'K[0].B[1][0].C[1]', 'K[0].B[1][0].C[1][0]', 'K[0].j', 'K[0].J', 'K[0].J[0]', 'K[0].J[1]', 'K[0].J[1][0]', 'K[0].J[1][0].b', 'K[0].J[1][0].b.v'])
        order = list((str(obj) for obj in pmo.preorder_traversal(model, descend=lambda x: x is not model.k and x is not model.K)))
        self.assertEqual(order, ['<block>', 'v', 'V', 'V[0]', 'V[1]', 'V[1][0]', 'c', 'C', 'C[0]', 'C[1]', 'C[1][0]', 'b', 'b.v', 'b.V', 'b.V[0]', 'b.V[1]', 'b.V[1][0]', 'b.c', 'b.C', 'b.C[0]', 'b.C[1]', 'b.C[1][0]', 'B', 'B[0]', 'B[0].v', 'B[0].V', 'B[0].V[0]', 'B[0].V[1]', 'B[0].V[1][0]', 'B[0].c', 'B[0].C', 'B[0].C[0]', 'B[0].C[1]', 'B[0].C[1][0]', 'B[1]', 'B[1][0]', 'B[1][0].v', 'B[1][0].V', 'B[1][0].V[0]', 'B[1][0].V[1]', 'B[1][0].V[1][0]', 'B[1][0].c', 'B[1][0].C', 'B[1][0].C[0]', 'B[1][0].C[1]', 'B[1][0].C[1][0]', 'j', 'J', 'J[0]', 'J[1]', 'J[1][0]', 'J[1][0].b', 'J[1][0].b.v', 'k', 'K'])
        order = list((str(obj) for obj in pmo.preorder_traversal(model, ctype=IBlock)))
        self.assertEqual(order, ['<block>', 'b', 'B', 'B[0]', 'B[1]', 'B[1][0]', 'j', 'J', 'J[0]', 'J[1]', 'J[1][0]', 'J[1][0].b', 'k', 'K', 'K[0]', 'K[0].b', 'K[0].B', 'K[0].B[0]', 'K[0].B[1]', 'K[0].B[1][0]', 'K[0].j', 'K[0].J', 'K[0].J[0]', 'K[0].J[1]', 'K[0].J[1][0]', 'K[0].J[1][0].b'])
        order = list((str(obj) for obj in pmo.preorder_traversal(model, ctype=IVariable)))
        self.assertEqual(order, ['<block>', 'v', 'V', 'V[0]', 'V[1]', 'V[1][0]', 'b', 'b.v', 'b.V', 'b.V[0]', 'b.V[1]', 'b.V[1][0]', 'B', 'B[0]', 'B[0].v', 'B[0].V', 'B[0].V[0]', 'B[0].V[1]', 'B[0].V[1][0]', 'B[1]', 'B[1][0]', 'B[1][0].v', 'B[1][0].V', 'B[1][0].V[0]', 'B[1][0].V[1]', 'B[1][0].V[1][0]', 'j', 'J', 'J[0]', 'J[1]', 'J[1][0]', 'J[1][0].b', 'J[1][0].b.v', 'k', 'K', 'K[0]', 'K[0].v', 'K[0].V', 'K[0].V[0]', 'K[0].V[1]', 'K[0].V[1][0]', 'K[0].b', 'K[0].b.v', 'K[0].b.V', 'K[0].b.V[0]', 'K[0].b.V[1]', 'K[0].b.V[1][0]', 'K[0].B', 'K[0].B[0]', 'K[0].B[0].v', 'K[0].B[0].V', 'K[0].B[0].V[0]', 'K[0].B[0].V[1]', 'K[0].B[0].V[1][0]', 'K[0].B[1]', 'K[0].B[1][0]', 'K[0].B[1][0].v', 'K[0].B[1][0].V', 'K[0].B[1][0].V[0]', 'K[0].B[1][0].V[1]', 'K[0].B[1][0].V[1][0]', 'K[0].j', 'K[0].J', 'K[0].J[0]', 'K[0].J[1]', 'K[0].J[1][0]', 'K[0].J[1][0].b', 'K[0].J[1][0].b.v'])

    def test_components(self):
        model = self.model.clone()
        checked = []

        def descend_into(x):
            self.assertTrue(x._is_heterogeneous_container)
            checked.append(x.name)
            return True
        order = list((str(obj) for obj in model.components(descend_into=descend_into)))
        self.assertEqual(checked, ['b', 'B[0]', 'B[1][0]', 'j', 'J[0]', 'J[1][0]', 'J[1][0].b', 'k', 'K[0]', 'K[0].b', 'K[0].B[0]', 'K[0].B[1][0]', 'K[0].j', 'K[0].J[0]', 'K[0].J[1][0]', 'K[0].J[1][0].b'])
        self.assertEqual(order, ['v', 'V[0]', 'V[1][0]', 'c', 'C[0]', 'C[1][0]', 'b', 'b.v', 'b.V[0]', 'b.V[1][0]', 'b.c', 'b.C[0]', 'b.C[1][0]', 'B[0]', 'B[0].v', 'B[0].V[0]', 'B[0].V[1][0]', 'B[0].c', 'B[0].C[0]', 'B[0].C[1][0]', 'B[1][0]', 'B[1][0].v', 'B[1][0].V[0]', 'B[1][0].V[1][0]', 'B[1][0].c', 'B[1][0].C[0]', 'B[1][0].C[1][0]', 'j', 'J[0]', 'J[1][0]', 'J[1][0].b', 'J[1][0].b.v', 'k', 'K[0]', 'K[0].v', 'K[0].V[0]', 'K[0].V[1][0]', 'K[0].c', 'K[0].C[0]', 'K[0].C[1][0]', 'K[0].b', 'K[0].b.v', 'K[0].b.V[0]', 'K[0].b.V[1][0]', 'K[0].b.c', 'K[0].b.C[0]', 'K[0].b.C[1][0]', 'K[0].B[0]', 'K[0].B[0].v', 'K[0].B[0].V[0]', 'K[0].B[0].V[1][0]', 'K[0].B[0].c', 'K[0].B[0].C[0]', 'K[0].B[0].C[1][0]', 'K[0].B[1][0]', 'K[0].B[1][0].v', 'K[0].B[1][0].V[0]', 'K[0].B[1][0].V[1][0]', 'K[0].B[1][0].c', 'K[0].B[1][0].C[0]', 'K[0].B[1][0].C[1][0]', 'K[0].j', 'K[0].J[0]', 'K[0].J[1][0]', 'K[0].J[1][0].b', 'K[0].J[1][0].b.v'])
        vlist = [str(obj) for obj in model.components(ctype=IVariable)]
        self.assertEqual(len(vlist), len(set(vlist)))
        clist = [str(obj) for obj in model.components(ctype=IConstraint)]
        self.assertEqual(len(clist), len(set(clist)))
        blist = [str(obj) for obj in model.components(ctype=IBlock)]
        self.assertEqual(len(blist), len(set(blist)))
        jlist = [str(obj) for obj in model.components(ctype=IJunk)]
        self.assertEqual(len(jlist), len(set(jlist)))
        for l1, l2 in itertools.product([vlist, clist, blist, jlist], repeat=2):
            if l1 is l2:
                continue
            self.assertEqual(set(l1).intersection(set(l2)), set([]))
        self.assertEqual(len(vlist) + len(clist) + len(blist) + len(jlist), len(order))

    def test_getname(self):
        model = self.model.clone()
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True), 'J[1][0].b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True, relative_to=model.J[1][0]), 'b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True, relative_to=model.J[1]), '[0].b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True, relative_to=model.J), '[1][0].b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True, relative_to=model), 'J[1][0].b.v')

    def test_heterogeneous_containers(self):
        order = list((str(obj) for obj in heterogeneous_containers(self.model.V)))
        self.assertEqual(order, [])
        order = list((str(obj) for obj in heterogeneous_containers(self.model.v)))
        self.assertEqual(order, [])
        order = list((str(obj) for obj in heterogeneous_containers(self.model)))
        self.assertEqual(order, ['<block>', 'b', 'B[0]', 'B[1][0]', 'k', 'K[0]', 'K[0].b', 'K[0].B[0]', 'K[0].B[1][0]', 'K[0].j', 'K[0].J[0]', 'K[0].J[1][0]', 'K[0].J[1][0].b', 'j', 'J[0]', 'J[1][0]', 'J[1][0].b'])

        def f(x):
            self.assertTrue(x._is_heterogeneous_container)
            parent = x.parent
            while parent is not None:
                if parent is self.model:
                    return False
                parent = parent.parent
            return True
        order1 = list((str(obj) for obj in heterogeneous_containers(self.model, descend_into=f)))
        order2 = list((str(obj) for obj in heterogeneous_containers(self.model, descend_into=lambda x: True if x is self.model else False)))
        self.assertEqual(order1, order2)
        self.assertEqual(order1, ['<block>', 'b', 'B[0]', 'B[1][0]', 'k', 'K[0]', 'j', 'J[0]', 'J[1][0]'])
        order = list((str(obj) for obj in heterogeneous_containers(self.model, ctype=IBlock)))
        self.assertEqual(order, ['<block>', 'b', 'B[0]', 'B[1][0]', 'k', 'K[0]', 'K[0].b', 'K[0].B[0]', 'K[0].B[1][0]', 'K[0].J[1][0].b', 'J[1][0].b'])
        order = list((str(obj) for obj in heterogeneous_containers(self.model, ctype=IJunk)))
        self.assertEqual(order, ['K[0].j', 'K[0].J[0]', 'K[0].J[1][0]', 'j', 'J[0]', 'J[1][0]'])
        order = list((str(obj) for obj in heterogeneous_containers(self.model.K, ctype=IJunk)))
        self.assertEqual(order, ['K[0].j', 'K[0].J[0]', 'K[0].J[1][0]'])
        order = list((str(obj) for obj in heterogeneous_containers(self.model.K[0], ctype=IJunk)))
        self.assertEqual(order, ['K[0].j', 'K[0].J[0]', 'K[0].J[1][0]'])
        order = list((str(obj) for obj in heterogeneous_containers(self.model.K[0].j, ctype=IJunk)))
        self.assertEqual(order, ['K[0].j'])
        order = list((str(obj) for obj in heterogeneous_containers(self.model.K[0].j, ctype=IBlock)))
        self.assertEqual(order, [])