import collections.abc
import pickle
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.list_container import ListContainer
from pyomo.core.kernel.block import block, block_list
class _TestActiveListContainerBase(_TestListContainerBase):

    def test_active_type(self):
        clist = self._container_type()
        self.assertTrue(isinstance(clist, ICategorizedObject))
        self.assertTrue(isinstance(clist, ICategorizedObjectContainer))
        self.assertTrue(isinstance(clist, IHomogeneousContainer))
        self.assertTrue(isinstance(clist, ListContainer))
        self.assertTrue(isinstance(clist, collections.abc.Sequence))
        self.assertTrue(issubclass(type(clist), collections.abc.Sequence))
        self.assertTrue(isinstance(clist, collections.abc.MutableSequence))
        self.assertTrue(issubclass(type(clist), collections.abc.MutableSequence))

    def test_active(self):
        index = list(range(4))
        clist = self._container_type((self._ctype_factory() for i in index))
        with self.assertRaises(AttributeError):
            clist.active = False
        for c in clist:
            with self.assertRaises(AttributeError):
                c.active = False
        model = block()
        model.clist = clist
        b = block()
        b.model = model
        blist = block_list()
        blist.append(b)
        blist.append(block())
        m = block()
        m.blist = blist
        self.assertEqual(m.active, True)
        self.assertEqual(blist.active, True)
        self.assertEqual(blist[1].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(clist.active, True)
        for c in clist:
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())), len(list(clist.components(active=True))))
        m.deactivate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for c in clist:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())), len(list(clist.components(active=None))))
        self.assertEqual(len(list(clist.components(active=True))), 0)
        test_c = clist[0]
        clist.remove(test_c)
        clist.append(test_c)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for c in clist:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())), len(list(clist.components(active=None))))
        self.assertEqual(len(list(clist.components(active=True))), 0)
        clist.remove(test_c)
        test_c.activate()
        self.assertEqual(test_c.active, True)
        self.assertEqual(clist.active, False)
        clist.append(test_c)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        clist.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for c in clist:
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components():
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(clist.components())), len(list(clist.components(active=None))))
        self.assertEqual(len(list(clist.components(active=True))), 1)
        m.activate(shallow=False)
        self.assertEqual(m.active, True)
        self.assertEqual(blist.active, True)
        self.assertEqual(blist[1].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(clist.active, True)
        for c in clist:
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())), len(list(clist.components(active=True))))
        m.deactivate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for c in clist:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())), len(list(clist.components(active=None))))
        self.assertEqual(len(list(clist.components(active=True))), 0)
        clist[len(clist) - 1] = self._ctype_factory()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        clist.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            if i == len(clist) - 1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for i, c in enumerate(clist.components(active=None)):
            if i == len(clist) - 1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(clist.components())), len(list(clist.components(active=None))))
        self.assertEqual(len(list(clist.components(active=True))), 1)
        clist.activate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())), len(list(clist.components(active=True))))
        clist.deactivate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for i, c in enumerate(clist):
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())), len(list(clist.components(active=None))))
        self.assertEqual(len(list(clist.components(active=True))), 0)
        clist[-1].activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        clist.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            if i == len(clist) - 1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for i, c in enumerate(clist.components(active=None)):
            if i == len(clist) - 1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(clist.components())), len(list(clist.components(active=None))))
        self.assertEqual(len(list(clist.components(active=True))), 1)
        clist.deactivate(shallow=False)
        clist.activate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())), len(list(clist.components(active=True))))

    def test_preorder_traversal(self):
        clist, traversal = super(_TestActiveListContainerBase, self).test_preorder_traversal()
        descend = lambda x: not x._is_heterogeneous_container
        clist[1].deactivate()
        self.assertEqual([None, '[0]', '[2]'], [c.name for c in pmo.preorder_traversal(clist, active=True, descend=descend)])
        self.assertEqual([id(clist), id(clist[0]), id(clist[2])], [id(c) for c in pmo.preorder_traversal(clist, active=True, descend=descend)])
        clist[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active], [c.name for c in pmo.preorder_traversal(clist, active=True, descend=descend)])
        self.assertEqual([id(c) for c in traversal if c.active], [id(c) for c in pmo.preorder_traversal(clist, active=True, descend=descend)])
        clist.deactivate()
        self.assertEqual(len(list(pmo.preorder_traversal(clist, active=True))), 0)
        self.assertEqual(len(list(pmo.generate_names(clist, active=True))), 0)

    def test_preorder_traversal_descend_check(self):
        clist, traversal = super(_TestActiveListContainerBase, self).test_preorder_traversal_descend_check()
        clist[1].deactivate()

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        order = list(pmo.preorder_traversal(clist, active=True, descend=descend))
        self.assertEqual([None, '[0]', '[2]'], [c.name for c in order])
        self.assertEqual([id(clist), id(clist[0]), id(clist[2])], [id(c) for c in order])
        if clist.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(clist), id(clist[0]), id(clist[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None], [c.name for c in descend.seen])
            self.assertEqual([id(clist)], [id(c) for c in descend.seen])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active and (not x._is_heterogeneous_container)
        descend.seen = []
        order = list(pmo.preorder_traversal(clist, active=None, descend=descend))
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
        self.assertEqual([id(clist), id(clist[0]), id(clist[1]), id(clist[2])], [id(c) for c in order])
        if clist.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(clist), id(clist[0]), id(clist[1]), id(clist[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
            self.assertEqual([id(clist), id(clist[1])], [id(c) for c in descend.seen])
        clist[1].deactivate(shallow=False)

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        order = list(pmo.preorder_traversal(clist, active=True, descend=descend))
        self.assertEqual([c.name for c in traversal if c.active], [c.name for c in order])
        self.assertEqual([id(c) for c in traversal if c.active], [id(c) for c in order])
        self.assertEqual([c.name for c in traversal if c.active and c._is_container], [c.name for c in descend.seen])
        self.assertEqual([id(c) for c in traversal if c.active and c._is_container], [id(c) for c in descend.seen])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active and (not x._is_heterogeneous_container)
        descend.seen = []
        order = list(pmo.preorder_traversal(clist, active=None, descend=descend))
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
        self.assertEqual([id(clist), id(clist[0]), id(clist[1]), id(clist[2])], [id(c) for c in order])
        if clist.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(clist), id(clist[0]), id(clist[1]), id(clist[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
            self.assertEqual([id(clist), id(clist[1])], [id(c) for c in descend.seen])
        clist.deactivate()

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return True
        descend.seen = []
        order = list(pmo.preorder_traversal(clist, active=True, descend=descend))
        self.assertEqual(len(descend.seen), 0)
        self.assertEqual(len(list(pmo.generate_names(clist, active=True))), 0)

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active
        descend.seen = []
        order = list(pmo.preorder_traversal(clist, active=None, descend=descend))
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], clist)