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
class _Test_block_base(object):
    _children = None
    _child_key = None
    _components_no_descend = None
    _components = None
    _blocks_no_descend = None
    _blocks = None
    _block = None

    def test_overwrite_warning(self):
        b = self._block.clone()
        name = 'x'
        while hasattr(b, name):
            name += 'x'
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            setattr(b, name, variable())
            setattr(b, name, getattr(b, name))
        assert out.getvalue() == '', str(out.getvalue())
        with LoggingIntercept(out, 'pyomo.core'):
            setattr(b, name, variable())
        assert out.getvalue() == 'Implicitly replacing attribute %s (type=variable) on block with new object (type=variable). This is usually indicative of a modeling error. To avoid this warning, delete the original object from the block before assigning a new object.\n' % name
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            setattr(b, name, 1.0)
        assert out.getvalue() == 'Implicitly replacing attribute %s (type=variable) on block with new object (type=float). This is usually indicative of a modeling error. To avoid this warning, delete the original object from the block before assigning a new object.\n' % name

    def test_clone(self):
        b = self._block
        bc = b.clone()
        self.assertIsNot(b, bc)
        self.assertEqual(len(list(b.children())), len(list(bc.children())))
        for c1, c2 in zip(b.children(), bc.children()):
            self.assertIs(c1.parent, b)
            self.assertIs(c2.parent, bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)
        self.assertEqual(len(list(b.components())), len(list(bc.components())))
        for c1, c2 in zip(b.components(), bc.components()):
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

    def test_pickle(self):
        b = pickle.loads(pickle.dumps(self._block))
        self.assertEqual(len(list(pmo.preorder_traversal(b, active=None))), len(self._names) + 1)
        names = pmo.generate_names(b, active=None)
        self.assertEqual(sorted(names.values()), sorted(self._names.values()))

    def test_preorder_traversal(self):
        self.assertEqual([str(obj) for obj in pmo.preorder_traversal(self._block, active=None)], [str(obj) for obj in self._preorder])
        self.assertEqual([id(obj) for obj in pmo.preorder_traversal(self._block, active=None)], [id(obj) for obj in self._preorder])
        self.assertEqual([str(obj) for obj in pmo.preorder_traversal(self._block, active=None, ctype=IVariable)], [str(obj) for obj in self._preorder if obj.ctype in (IBlock, IVariable)])
        self.assertEqual([id(obj) for obj in pmo.preorder_traversal(self._block, active=None, ctype=IVariable)], [id(obj) for obj in self._preorder if obj.ctype in (IBlock, IVariable)])

    def test_preorder_traversal_descend_check(self):

        def descend(x):
            self.assertTrue(x._is_container)
            return True
        order = list(pmo.preorder_traversal(self._block, active=None, descend=descend))
        self.assertEqual([str(obj) for obj in order], [str(obj) for obj in self._preorder])
        self.assertEqual([id(obj) for obj in order], [id(obj) for obj in self._preorder])

        def descend(x):
            self.assertTrue(x._is_container)
            return True
        order = list(pmo.preorder_traversal(self._block, active=None, ctype=IVariable, descend=descend))
        self.assertEqual([str(obj) for obj in order], [str(obj) for obj in self._preorder if obj.ctype in (IBlock, IVariable)])
        self.assertEqual([id(obj) for obj in order], [id(obj) for obj in self._preorder if obj.ctype in (IBlock, IVariable)])

        def descend(x):
            if x.parent is self._block:
                return False
            return True
        order = list(pmo.preorder_traversal(self._block, active=None, descend=descend))
        self.assertEqual([str(obj) for obj in order], [str(obj) for obj in self._preorder if obj.parent is None or obj.parent is self._block])
        self.assertEqual([id(obj) for obj in order], [id(obj) for obj in self._preorder if obj.parent is None or obj.parent is self._block])

    def test_child(self):
        for child in self._child_key:
            parent = child.parent
            self.assertTrue(parent is not None)
            self.assertTrue(id(child) in set((id(_c) for _c in self._children[parent])))
            self.assertIs(parent.child(self._child_key[child]), child)
            with self.assertRaises(KeyError):
                parent.child('_not_a_valid_child_key_')

    def test_children(self):
        for obj in self._children:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            if isinstance(obj, IBlock):
                for child in obj.children():
                    self.assertTrue(child.parent is obj)
                self.assertEqual(sorted((str(child) for child in obj.children())), sorted((str(child) for child in self._children[obj])))
                self.assertEqual(set((id(child) for child in obj.children())), set((id(child) for child in self._children[obj])))
                self.assertEqual(sorted((str(child) for child in obj.children(ctype=IBlock))), sorted((str(child) for child in self._children[obj] if child.ctype is IBlock)))
                self.assertEqual(set((id(child) for child in obj.children(ctype=IBlock))), set((id(child) for child in self._children[obj] if child.ctype is IBlock)))
                self.assertEqual(sorted((str(child) for child in obj.children(ctype=IVariable))), sorted((str(child) for child in self._children[obj] if child.ctype is IVariable)))
                self.assertEqual(set((id(child) for child in obj.children(ctype=IVariable))), set((id(child) for child in self._children[obj] if child.ctype is IVariable)))
            elif isinstance(obj, ICategorizedObjectContainer):
                for child in obj.children():
                    self.assertTrue(child.parent is obj)
                self.assertEqual(set((id(child) for child in obj.children())), set((id(child) for child in self._children[obj])))
            else:
                self.assertEqual(len(self._children[obj]), 0)

    def test_components_no_descend_active_None(self):
        for obj in self._components_no_descend:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            for c in obj.components(descend_into=False):
                self.assertTrue(_path_to_object_exists(obj, c))
            self.assertEqual(sorted((str(_b) for _b in obj.components(ctype=IBlock, active=None, descend_into=False))), sorted((str(_b) for _b in self._components_no_descend[obj][IBlock])))
            self.assertEqual(set((id(_b) for _b in obj.components(ctype=IBlock, active=None, descend_into=False))), set((id(_b) for _b in self._components_no_descend[obj][IBlock])))
            self.assertEqual(sorted((str(_v) for _v in obj.components(ctype=IVariable, active=None, descend_into=False))), sorted((str(_v) for _v in self._components_no_descend[obj][IVariable])))
            self.assertEqual(set((id(_v) for _v in obj.components(ctype=IVariable, active=None, descend_into=False))), set((id(_v) for _v in self._components_no_descend[obj][IVariable])))
            self.assertEqual(sorted((str(_c) for _c in obj.components(active=None, descend_into=False))), sorted((str(_c) for ctype in self._components_no_descend[obj] for _c in self._components_no_descend[obj][ctype])))
            self.assertEqual(set((id(_c) for _c in obj.components(active=None, descend_into=False))), set((id(_c) for ctype in self._components_no_descend[obj] for _c in self._components_no_descend[obj][ctype])))

    def test_components_no_descend_active_True(self):
        for obj in self._components_no_descend:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            self.assertEqual(sorted((str(_b) for _b in obj.components(ctype=IBlock, active=True, descend_into=False))), sorted((str(_b) for _b in self._components_no_descend[obj][IBlock] if _b.active)) if getattr(obj, 'active', True) else [])
            self.assertEqual(set((id(_b) for _b in obj.components(ctype=IBlock, active=True, descend_into=False))), set((id(_b) for _b in self._components_no_descend[obj][IBlock] if _b.active)) if getattr(obj, 'active', True) else set())
            self.assertEqual(sorted((str(_v) for _v in obj.components(ctype=IVariable, active=True, descend_into=False))), sorted((str(_v) for _v in self._components_no_descend[obj][IVariable])) if getattr(obj, 'active', True) else [])
            self.assertEqual(set((id(_v) for _v in obj.components(ctype=IVariable, active=True, descend_into=False))), set((id(_v) for _v in self._components_no_descend[obj][IVariable])) if getattr(obj, 'active', True) else set())
            self.assertEqual(sorted((str(_c) for _c in obj.components(active=True, descend_into=False))), sorted((str(_c) for ctype in self._components_no_descend[obj] for _c in self._components_no_descend[obj][ctype] if getattr(_c, 'active', True))) if getattr(obj, 'active', True) else [])
            self.assertEqual(set((id(_c) for _c in obj.components(active=True, descend_into=False))), set((id(_c) for ctype in self._components_no_descend[obj] for _c in self._components_no_descend[obj][ctype] if getattr(_c, 'active', True))) if getattr(obj, 'active', True) else set())

    def test_components_active_None(self):
        for obj in self._components:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            for c in obj.components(descend_into=True):
                self.assertTrue(_path_to_object_exists(obj, c))
            self.assertEqual(sorted((str(_b) for _b in obj.components(ctype=IBlock, active=None, descend_into=True))), sorted((str(_b) for _b in self._components[obj][IBlock])))
            self.assertEqual(set((id(_b) for _b in obj.components(ctype=IBlock, active=None, descend_into=True))), set((id(_b) for _b in self._components[obj][IBlock])))
            self.assertEqual(sorted((str(_v) for _v in obj.components(ctype=IVariable, active=None, descend_into=True))), sorted((str(_v) for _v in self._components[obj][IVariable])))
            self.assertEqual(set((id(_v) for _v in obj.components(ctype=IVariable, active=None, descend_into=True))), set((id(_v) for _v in self._components[obj][IVariable])))
            self.assertEqual(sorted((str(_c) for _c in obj.components(active=None, descend_into=True))), sorted((str(_c) for ctype in self._components[obj] for _c in self._components[obj][ctype])))
            self.assertEqual(set((id(_c) for _c in obj.components(active=None, descend_into=True))), set((id(_c) for ctype in self._components[obj] for _c in self._components[obj][ctype])))

    def test_components_active_True(self):
        for obj in self._components:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            self.assertEqual(sorted((str(_b) for _b in obj.components(ctype=IBlock, active=True, descend_into=True))), sorted((str(_b) for _b in self._components[obj][IBlock] if _b.active)) if getattr(obj, 'active', True) else [])
            self.assertEqual(set((id(_b) for _b in obj.components(ctype=IBlock, active=True, descend_into=True))), set((id(_b) for _b in self._components[obj][IBlock] if _b.active)) if getattr(obj, 'active', True) else set())
            self.assertEqual(sorted((str(_v) for _v in obj.components(ctype=IVariable, active=True, descend_into=True))), sorted((str(_v) for _v in self._components[obj][IVariable] if _active_path_to_object_exists(obj, _v))) if getattr(obj, 'active', True) else [])
            self.assertEqual(set((id(_v) for _v in obj.components(ctype=IVariable, active=True, descend_into=True))), set((id(_v) for _v in self._components[obj][IVariable] if _active_path_to_object_exists(obj, _v))) if getattr(obj, 'active', True) else set())
            self.assertEqual(sorted((str(_c) for _c in obj.components(active=True, descend_into=True))), sorted((str(_c) for ctype in self._components[obj] for _c in self._components[obj][ctype] if _active_path_to_object_exists(obj, _c))) if getattr(obj, 'active', True) else [])
            self.assertEqual(set((id(_c) for _c in obj.components(active=True, descend_into=True))), set((id(_c) for ctype in self._components[obj] for _c in self._components[obj][ctype] if _active_path_to_object_exists(obj, _c))) if getattr(obj, 'active', True) else set())