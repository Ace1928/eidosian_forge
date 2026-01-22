import collections.abc
import pickle
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.block import block, block_dict
class _TestActiveDictContainerBase(_TestDictContainerBase):

    def test_active_type(self):
        cdict = self._container_type()
        self.assertTrue(isinstance(cdict, ICategorizedObject))
        self.assertTrue(isinstance(cdict, ICategorizedObjectContainer))
        self.assertTrue(isinstance(cdict, IHomogeneousContainer))
        self.assertTrue(isinstance(cdict, DictContainer))
        self.assertTrue(isinstance(cdict, collections.abc.Mapping))
        self.assertTrue(isinstance(cdict, collections.abc.MutableMapping))

    def test_active(self):
        children = {}
        children['a'] = self._ctype_factory()
        children[1] = self._ctype_factory()
        children[None] = self._ctype_factory()
        children[1,] = self._ctype_factory()
        children[1, 2] = self._ctype_factory()
        children['(1,2)'] = self._ctype_factory()
        cdict = self._container_type()
        cdict.update(children)
        with self.assertRaises(AttributeError):
            cdict.active = False
        for c in cdict.values():
            with self.assertRaises(AttributeError):
                c.active = False
        model = block()
        model.cdict = cdict
        b = block()
        b.model = model
        bdict = block_dict()
        bdict[0] = b
        bdict[None] = block()
        m = block()
        m.bdict = bdict
        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)
        for c in cdict.components():
            self.assertEqual(c.active, True)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(cdict.components())), len(cdict))
        self.assertEqual(len(list(cdict.components())), len(list(cdict.components(active=True))))
        m.deactivate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(cdict.components())), len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 0)
        test_key = list(children.keys())[0]
        del cdict[test_key]
        cdict[test_key] = children[test_key]
        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(cdict.components())), len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 0)
        del cdict[test_key]
        children[test_key].activate()
        self.assertEqual(children[test_key].active, True)
        cdict[test_key] = children[test_key]
        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, False)
        cdict.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, True)
        for key, c in cdict.items():
            if key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in cdict.components():
            if c.storage_key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(cdict.components())), len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 1)
        cdict.deactivate()
        m.activate(shallow=False)
        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)
        for c in cdict.components():
            self.assertEqual(c.active, True)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(cdict.components())), len(cdict))
        self.assertEqual(len(list(cdict.components())), len(list(cdict.components(active=True))))
        cdict.deactivate(shallow=False)
        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(cdict.components())), len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 0)
        cdict.activate(shallow=False)
        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)
        for c in cdict.components():
            self.assertEqual(c.active, True)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(cdict.components())), len(cdict))
        self.assertEqual(len(list(cdict.components())), len(list(cdict.components(active=True))))
        cdict.deactivate(shallow=False)
        cdict[test_key].activate()
        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, False)
        cdict.activate()
        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for key, c in cdict.items():
            if key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in cdict.components():
            if c.storage_key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(cdict.components())), len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 1)

    def test_preorder_traversal(self):
        cdict, traversal = super(_TestActiveDictContainerBase, self).test_preorder_traversal()
        descend = lambda x: not x._is_heterogeneous_container
        cdict[1].deactivate()
        self.assertEqual([None, '[0]', '[2]'], [c.name for c in pmo.preorder_traversal(cdict, active=True, descend=descend)])
        self.assertEqual([id(cdict), id(cdict[0]), id(cdict[2])], [id(c) for c in pmo.preorder_traversal(cdict, active=True, descend=descend)])
        cdict[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active], [c.name for c in pmo.preorder_traversal(cdict, active=True, descend=descend)])
        self.assertEqual([id(c) for c in traversal if c.active], [id(c) for c in pmo.preorder_traversal(cdict, active=True, descend=descend)])
        cdict.deactivate()
        self.assertEqual(len(list(pmo.preorder_traversal(cdict, active=True))), 0)
        self.assertEqual(len(list(pmo.generate_names(cdict, active=True))), 0)

    def test_preorder_traversal_descend_check(self):
        cdict, traversal = super(_TestActiveDictContainerBase, self).test_preorder_traversal_descend_check()
        cdict[1].deactivate()

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict, active=True, descend=descend))
        self.assertEqual([None, '[0]', '[2]'], [c.name for c in order])
        self.assertEqual([id(cdict), id(cdict[0]), id(cdict[2])], [id(c) for c in order])
        if cdict.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(cdict), id(cdict[0]), id(cdict[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None], [c.name for c in descend.seen])
            self.assertEqual([id(cdict)], [id(c) for c in descend.seen])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active and (not x._is_heterogeneous_container)
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict, active=None, descend=descend))
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
        self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in order])
        if cdict.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
            self.assertEqual([id(cdict), id(cdict[1])], [id(c) for c in descend.seen])
        cdict[1].deactivate(shallow=False)

        def descend(x):
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict, active=True, descend=descend))
        self.assertEqual([c.name for c in traversal if c.active], [c.name for c in order])
        self.assertEqual([id(c) for c in traversal if c.active], [id(c) for c in order])
        self.assertEqual([c.name for c in traversal if c.active and c._is_container], [c.name for c in descend.seen])
        self.assertEqual([id(c) for c in traversal if c.active and c._is_container], [id(c) for c in descend.seen])

        def descend(x):
            descend.seen.append(x)
            return x.active and (not x._is_heterogeneous_container)
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict, active=None, descend=descend))
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
        self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in order])
        if cdict.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
            self.assertEqual([id(cdict), id(cdict[1])], [id(c) for c in descend.seen])
        cdict.deactivate()

        def descend(x):
            descend.seen.append(x)
            return True
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict, active=True, descend=descend))
        self.assertEqual(len(descend.seen), 0)
        self.assertEqual(len(list(pmo.generate_names(cdict, active=True))), 0)

        def descend(x):
            descend.seen.append(x)
            return x.active
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict, active=None, descend=descend))
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], cdict)