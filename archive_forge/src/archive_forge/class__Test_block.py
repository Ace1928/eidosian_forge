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
class _Test_block(_Test_block_base):
    _do_clone = None

    @classmethod
    def setUpClass(cls):
        assert cls._do_clone is not None
        model = cls._block = block()
        model.v_1 = variable()
        model.vdict_1 = variable_dict()
        model.vdict_1[None] = variable()
        model.vlist_1 = variable_list()
        model.vlist_1.append(variable())
        model.vlist_1.append(variable())
        model.b_1 = block()
        model.b_1.v_2 = variable()
        model.b_1.b_2 = block()
        model.b_1.b_2.b_3 = block()
        model.b_1.b_2.v_3 = variable()
        model.b_1.b_2.vlist_3 = variable_list()
        model.b_1.b_2.vlist_3.append(variable())
        model.b_1.b_2.deactivate(shallow=False)
        model.bdict_1 = block_dict()
        model.blist_1 = block_list()
        model.blist_1.append(block())
        model.blist_1[0].v_2 = variable()
        model.blist_1[0].b_2 = block()
        if cls._do_clone:
            model = cls._block = model.clone()
        cls._preorder = [model, model.v_1, model.vdict_1, model.vdict_1[None], model.vlist_1, model.vlist_1[0], model.vlist_1[1], model.b_1, model.b_1.v_2, model.b_1.b_2, model.b_1.b_2.b_3, model.b_1.b_2.v_3, model.b_1.b_2.vlist_3, model.b_1.b_2.vlist_3[0], model.bdict_1, model.blist_1, model.blist_1[0], model.blist_1[0].v_2, model.blist_1[0].b_2]
        cls._names = ComponentMap()
        cls._names[model.v_1] = 'v_1'
        cls._names[model.vdict_1] = 'vdict_1'
        cls._names[model.vdict_1[None]] = 'vdict_1[None]'
        cls._names[model.vlist_1] = 'vlist_1'
        cls._names[model.vlist_1[0]] = 'vlist_1[0]'
        cls._names[model.vlist_1[1]] = 'vlist_1[1]'
        cls._names[model.b_1] = 'b_1'
        cls._names[model.b_1.v_2] = 'b_1.v_2'
        cls._names[model.b_1.b_2] = 'b_1.b_2'
        cls._names[model.b_1.b_2.b_3] = 'b_1.b_2.b_3'
        cls._names[model.b_1.b_2.v_3] = 'b_1.b_2.v_3'
        cls._names[model.b_1.b_2.vlist_3] = 'b_1.b_2.vlist_3'
        cls._names[model.b_1.b_2.vlist_3[0]] = 'b_1.b_2.vlist_3[0]'
        cls._names[model.bdict_1] = 'bdict_1'
        cls._names[model.blist_1] = 'blist_1'
        cls._names[model.blist_1[0]] = 'blist_1[0]'
        cls._names[model.blist_1[0].v_2] = 'blist_1[0].v_2'
        cls._names[model.blist_1[0].b_2] = 'blist_1[0].b_2'
        cls._children = ComponentMap()
        cls._children[model] = [model.v_1, model.vdict_1, model.vlist_1, model.b_1, model.bdict_1, model.blist_1]
        cls._children[model.vdict_1] = [model.vdict_1[None]]
        cls._children[model.vlist_1] = [model.vlist_1[0], model.vlist_1[1]]
        cls._children[model.b_1] = [model.b_1.v_2, model.b_1.b_2]
        cls._children[model.b_1.b_2] = [model.b_1.b_2.v_3, model.b_1.b_2.vlist_3, model.b_1.b_2.b_3]
        cls._children[model.b_1.b_2.b_3] = []
        cls._children[model.b_1.b_2.vlist_3] = [model.b_1.b_2.vlist_3[0]]
        cls._children[model.bdict_1] = []
        cls._children[model.blist_1] = [model.blist_1[0]]
        cls._children[model.blist_1[0]] = [model.blist_1[0].v_2, model.blist_1[0].b_2]
        cls._child_key = ComponentMap()
        cls._child_key[model.v_1] = 'v_1'
        cls._child_key[model.vdict_1] = 'vdict_1'
        cls._child_key[model.vlist_1] = 'vlist_1'
        cls._child_key[model.b_1] = 'b_1'
        cls._child_key[model.bdict_1] = 'bdict_1'
        cls._child_key[model.blist_1] = 'blist_1'
        cls._child_key[model.vdict_1[None]] = None
        cls._child_key[model.vlist_1[0]] = 0
        cls._child_key[model.vlist_1[1]] = 1
        cls._child_key[model.b_1.v_2] = 'v_2'
        cls._child_key[model.b_1.b_2] = 'b_2'
        cls._child_key[model.b_1.b_2.b_3] = 'b_3'
        cls._child_key[model.b_1.b_2.v_3] = 'v_3'
        cls._child_key[model.b_1.b_2.vlist_3] = 'vlist_3'
        cls._child_key[model.b_1.b_2.vlist_3[0]] = 0
        cls._child_key[model.blist_1[0]] = 0
        cls._child_key[model.blist_1[0].v_2] = 'v_2'
        cls._child_key[model.blist_1[0].b_2] = 'b_2'
        cls._components_no_descend = ComponentMap()
        cls._components_no_descend[model] = {}
        cls._components_no_descend[model][IVariable] = [model.v_1, model.vdict_1[None], model.vlist_1[0], model.vlist_1[1]]
        cls._components_no_descend[model][IBlock] = [model.b_1, model.blist_1[0]]
        cls._components_no_descend[model.b_1] = {}
        cls._components_no_descend[model.b_1][IVariable] = [model.b_1.v_2]
        cls._components_no_descend[model.b_1][IBlock] = [model.b_1.b_2]
        cls._components_no_descend[model.b_1.b_2] = {}
        cls._components_no_descend[model.b_1.b_2][IVariable] = [model.b_1.b_2.v_3, model.b_1.b_2.vlist_3[0]]
        cls._components_no_descend[model.b_1.b_2][IBlock] = [model.b_1.b_2.b_3]
        cls._components_no_descend[model.b_1.b_2.b_3] = {}
        cls._components_no_descend[model.b_1.b_2.b_3][IVariable] = []
        cls._components_no_descend[model.b_1.b_2.b_3][IBlock] = []
        cls._components_no_descend[model.blist_1[0]] = {}
        cls._components_no_descend[model.blist_1[0]][IVariable] = [model.blist_1[0].v_2]
        cls._components_no_descend[model.blist_1[0]][IBlock] = [model.blist_1[0].b_2]
        cls._components_no_descend[model.blist_1[0].b_2] = {}
        cls._components_no_descend[model.blist_1[0].b_2][IVariable] = []
        cls._components_no_descend[model.blist_1[0].b_2][IBlock] = []
        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][IVariable] = [model.v_1, model.vdict_1[None], model.vlist_1[0], model.vlist_1[1], model.b_1.v_2, model.b_1.b_2.v_3, model.b_1.b_2.vlist_3[0], model.blist_1[0].v_2]
        cls._components[model][IBlock] = [model.b_1, model.blist_1[0], model.b_1.b_2, model.b_1.b_2.b_3, model.blist_1[0].b_2]
        cls._components[model.b_1] = {}
        cls._components[model.b_1][IVariable] = [model.b_1.v_2, model.b_1.b_2.v_3, model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1][IBlock] = [model.b_1.b_2, model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2] = {}
        cls._components[model.b_1.b_2][IVariable] = [model.b_1.b_2.v_3, model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1.b_2][IBlock] = [model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2.b_3] = {}
        cls._components[model.b_1.b_2.b_3][IVariable] = []
        cls._components[model.b_1.b_2.b_3][IBlock] = []
        cls._components[model.blist_1[0]] = {}
        cls._components[model.blist_1[0]][IVariable] = [model.blist_1[0].v_2]
        cls._components[model.blist_1[0]][IBlock] = [model.blist_1[0].b_2]
        cls._components[model.blist_1[0].b_2] = {}
        cls._components[model.blist_1[0].b_2][IVariable] = []
        cls._components[model.blist_1[0].b_2][IBlock] = []
        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = [obj] + cls._components_no_descend[obj][IBlock]
        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = [obj] + cls._components[obj][IBlock]

    def test_init(self):
        b = block()
        self.assertTrue(b.parent is None)
        self.assertEqual(b.ctype, IBlock)

    def test_type(self):
        b = block()
        self.assertTrue(isinstance(b, ICategorizedObject))
        self.assertTrue(isinstance(b, ICategorizedObjectContainer))
        self.assertTrue(isinstance(b, IHeterogeneousContainer))
        self.assertTrue(isinstance(b, IBlock))

    def test_overwrite(self):
        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = variable()
        self.assertTrue(v.parent is None)
        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = v
        self.assertTrue(v.parent is b)
        b = block()
        c = b.c = constraint()
        self.assertIs(c.parent, b)
        b.c = constraint()
        self.assertTrue(c.parent is None)
        b = block()
        c = b.c = constraint()
        self.assertIs(c.parent, b)
        b.c = c
        self.assertTrue(c.parent is b)
        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = constraint()
        self.assertTrue(v.parent is None)
        b = block()
        c = b.c = variable()
        self.assertIs(c.parent, b)
        b.c = variable()
        self.assertTrue(c.parent is None)

    def test_already_has_parent(self):
        b1 = block()
        v = b1.v = variable()
        b2 = block()
        with self.assertRaises(ValueError):
            b2.v = v
        self.assertTrue(v.parent is b1)
        del b1.v
        b2.v = v
        self.assertTrue(v.parent is b2)

    def test_delattr(self):
        b = block()
        with self.assertRaises(AttributeError):
            del b.not_an_attribute
        c = b.b = block()
        self.assertIs(c.parent, b)
        del b.b
        self.assertIs(c.parent, None)
        b.b = c
        self.assertIs(c.parent, b)
        b.x = 2
        self.assertTrue(hasattr(b, 'x'))
        self.assertEqual(b.x, 2)
        del b.x
        self.assertTrue(not hasattr(b, 'x'))

    def test_collect_ctypes_small_block_storage(self):
        b = block()
        self.assertEqual(b.collect_ctypes(active=None), set())
        self.assertEqual(b.collect_ctypes(), set())
        self.assertEqual(b.collect_ctypes(active=True), set())
        b.x = variable()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
        self.assertEqual(b.collect_ctypes(), set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
        b.y = constraint()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(), set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable, IConstraint]))
        b.y.deactivate()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(), set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
        B = block()
        B.b = b
        self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False), set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([IBlock]))
        self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(), set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(active=True), set([IBlock, IVariable]))
        b.deactivate()
        self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False), set([]))
        self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([]))
        self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(), set([]))
        self.assertEqual(B.collect_ctypes(active=True), set([]))
        B.x = variable()
        self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False), set([IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(), set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=True), set([IVariable]))
        del b.y
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
        self.assertEqual(b.collect_ctypes(), set([]))
        self.assertEqual(b.collect_ctypes(active=True), set([]))
        b.activate()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
        self.assertEqual(b.collect_ctypes(), set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
        del b.x
        self.assertEqual(b.collect_ctypes(), set())

    def test_collect_ctypes_large_block_storage(self):
        b = block()
        b._activate_large_storage_mode()
        self.assertEqual(b.collect_ctypes(active=None), set())
        self.assertEqual(b.collect_ctypes(), set())
        self.assertEqual(b.collect_ctypes(active=True), set())
        b.x = variable()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
        self.assertEqual(b.collect_ctypes(), set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
        b.y = constraint()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(), set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable, IConstraint]))
        b.y.deactivate()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(), set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
        B = block()
        b._activate_large_storage_mode()
        B.b = b
        self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False), set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([IBlock]))
        self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(), set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(active=True), set([IBlock, IVariable]))
        b.deactivate()
        self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False), set([]))
        self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([]))
        self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(), set([]))
        self.assertEqual(B.collect_ctypes(active=True), set([]))
        B.x = variable()
        self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False), set([IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(), set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=True), set([IVariable]))
        del b.y
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
        self.assertEqual(b.collect_ctypes(), set([]))
        self.assertEqual(b.collect_ctypes(active=True), set([]))
        b.activate()
        self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
        self.assertEqual(b.collect_ctypes(), set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
        del b.x
        self.assertEqual(b.collect_ctypes(), set())