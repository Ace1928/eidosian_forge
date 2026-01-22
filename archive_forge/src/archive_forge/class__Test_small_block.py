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
class _Test_small_block(_Test_block_base):
    _do_clone = None

    @classmethod
    def setUpClass(cls):
        assert cls._do_clone is not None
        cls._myblock_type = _MyBlock
        model = cls._block = _MyBlock()
        if cls._do_clone:
            model = cls._block = model.clone()
        cls._preorder = [model, model.b, model.b.v, model.b.blist, model.b.blist[0], model.v]
        cls._names = ComponentMap()
        cls._names[model.b] = 'b'
        cls._names[model.b.v] = 'b.v'
        cls._names[model.b.blist] = 'b.blist'
        cls._names[model.b.blist[0]] = 'b.blist[0]'
        cls._names[model.v] = 'v'
        cls._children = ComponentMap()
        cls._children[model] = [model.b, model.v]
        cls._children[model.b] = [model.b.v, model.b.blist]
        cls._children[model.b.blist] = [model.b.blist[0]]
        cls._children[model.b.blist[0]] = []
        cls._child_key = ComponentMap()
        cls._child_key[model.b] = 'b'
        cls._child_key[model.b.v] = 'v'
        cls._child_key[model.b.blist] = 'blist'
        cls._child_key[model.b.blist[0]] = 0
        cls._child_key[model.v] = 'v'
        cls._components_no_descend = ComponentMap()
        cls._components_no_descend[model] = {}
        cls._components_no_descend[model][IVariable] = [model.v]
        cls._components_no_descend[model][IBlock] = [model.b]
        cls._components_no_descend[model.b] = {}
        cls._components_no_descend[model.b][IVariable] = [model.b.v]
        cls._components_no_descend[model.b][IBlock] = [model.b.blist[0]]
        cls._components_no_descend[model.b.blist[0]] = {}
        cls._components_no_descend[model.b.blist[0]][IBlock] = []
        cls._components_no_descend[model.b.blist[0]][IVariable] = []
        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][IVariable] = [model.v, model.b.v]
        cls._components[model][IBlock] = [model.b, model.b.blist[0]]
        cls._components[model.b] = {}
        cls._components[model.b][IVariable] = [model.b.v]
        cls._components[model.b][IBlock] = [model.b.blist[0]]
        cls._components[model.b.blist[0]] = {}
        cls._components[model.b.blist[0]][IBlock] = []
        cls._components[model.b.blist[0]][IVariable] = []
        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = [obj] + cls._components_no_descend[obj][IBlock]
        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = [obj] + cls._components[obj][IBlock]

    def test_collect_ctypes(self):
        self.assertEqual(self._block.collect_ctypes(active=None), set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(), set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(active=True), set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(descend_into=False), set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(active=True, descend_into=False), set([IBlock, IVariable]))
        self._block.b.deactivate()
        try:
            self.assertEqual(self._block.collect_ctypes(active=None), set([IBlock, IVariable]))
            self.assertEqual(self._block.collect_ctypes(), set([IVariable]))
            self.assertEqual(self._block.collect_ctypes(active=True), set([IVariable]))
            self.assertEqual(self._block.collect_ctypes(active=None, descend_into=False), set([IBlock, IVariable]))
            self.assertEqual(self._block.collect_ctypes(descend_into=False), set([IVariable]))
            self.assertEqual(self._block.collect_ctypes(active=True, descend_into=False), set([IVariable]))
        finally:
            self._block.b.activate()

    def test_customblock_delattr(self):
        b = _MyBlock()
        with self.assertRaises(AttributeError):
            del b.not_an_attribute
        c = b.b
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

    def test_customblock_setattr(self):
        b = _MyBlockBase()
        self.assertIs(b.b.parent, b)
        self.assertIs(b.b.v.parent, b.b)
        with self.assertRaises(ValueError):
            b.b = b.b.v
        self.assertIs(b.b.parent, b)
        self.assertIs(b.b.v.parent, b.b)
        c = b.b
        self.assertIs(c.parent, b)
        b.b = c
        self.assertIs(c.parent, b)
        assert not hasattr(b, 'g')
        with self.assertRaises(ValueError):
            b.g = b.b
        self.assertIs(b.b.parent, b)
        b.g = 1
        with self.assertRaises(ValueError):
            b.g = b.b
        self.assertEqual(b.g, 1)
        self.assertIs(b.b.parent, b)
        b.b = block()
        self.assertIs(c.parent, None)
        self.assertIs(b.b.parent, b)

    def test_customblock__with_dict_setattr(self):
        b = _MyBlock()
        self.assertIs(b.v.parent, b)
        self.assertIs(b.b.parent, b)
        with self.assertRaises(ValueError):
            b.v = b.b
        self.assertIs(b.v.parent, b)
        self.assertIs(b.b.parent, b)
        b.not_an_attribute = 2
        v = b.v
        self.assertIs(v.parent, b)
        b.v = v
        self.assertIs(v.parent, b)
        b.v = variable()
        self.assertIs(v.parent, None)
        self.assertIs(b.v.parent, b)

    def test_inactive_behavior(self):
        b = _MyBlock()
        b.deactivate()
        self.assertNotEqual(len(list(pmo.preorder_traversal(b, active=None))), 0)
        self.assertEqual(len(list(pmo.preorder_traversal(b))), 0)
        self.assertEqual(len(list(pmo.preorder_traversal(b, active=True))), 0)

        def descend(x):
            return True
        self.assertNotEqual(len(list(pmo.preorder_traversal(b, active=None, descend=descend))), 0)
        self.assertEqual(len(list(pmo.preorder_traversal(b, descend=descend))), 0)
        self.assertEqual(len(list(pmo.preorder_traversal(b, active=True, descend=descend))), 0)

        def descend(x):
            descend.seen.append(x)
            return x.active
        descend.seen = []
        self.assertEqual(len(list(pmo.preorder_traversal(b, active=None, descend=descend))), 1)
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], b)
        self.assertNotEqual(len(list(b.components(active=None))), 0)
        self.assertEqual(len(list(b.components())), 0)
        self.assertEqual(len(list(b.components(active=True))), 0)
        self.assertNotEqual(len(list(pmo.generate_names(b, active=None))), 0)
        self.assertEqual(len(list(pmo.generate_names(b))), 0)
        self.assertEqual(len(list(pmo.generate_names(b, active=True))), 0)