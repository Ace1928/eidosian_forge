import collections.abc
import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.tuple_container import TupleContainer
from pyomo.core.kernel.block import block, block_list
class _TestActiveTupleContainerBase(_TestTupleContainerBase):

    def test_active_type(self):
        ctuple = self._container_type()
        self.assertTrue(isinstance(ctuple, ICategorizedObject))
        self.assertTrue(isinstance(ctuple, ICategorizedObjectContainer))
        self.assertTrue(isinstance(ctuple, IHomogeneousContainer))
        self.assertTrue(isinstance(ctuple, TupleContainer))
        self.assertTrue(isinstance(ctuple, collections.abc.Sequence))
        self.assertTrue(issubclass(type(ctuple), collections.abc.Sequence))

    def test_active(self):
        index = list(range(4))
        ctuple = self._container_type((self._ctype_factory() for i in index))
        with self.assertRaises(AttributeError):
            ctuple.active = False
        for c in ctuple:
            with self.assertRaises(AttributeError):
                c.active = False
        model = block()
        model.ctuple = ctuple
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
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())), len(list(ctuple.components(active=True))))
        m.deactivate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        for c in ctuple:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(ctuple.components())), len(list(ctuple.components(active=None))))
        self.assertEqual(len(list(ctuple.components(active=True))), 0)
        test_c = ctuple[0]
        test_c.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        ctuple.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in ctuple.components():
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(ctuple.components())), len(list(ctuple.components(active=None))))
        self.assertEqual(len(list(ctuple.components(active=True))), 1)
        m.activate(shallow=False)
        self.assertEqual(m.active, True)
        self.assertEqual(blist.active, True)
        self.assertEqual(blist[1].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())), len(list(ctuple.components(active=True))))
        m.deactivate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        for c in ctuple:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(ctuple.components())), len(list(ctuple.components(active=None))))
        self.assertEqual(len(list(ctuple.components(active=True))), 0)
        ctuple.activate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for i, c in enumerate(ctuple):
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())), len(list(ctuple.components(active=True))))
        ctuple.deactivate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        for i, c in enumerate(ctuple):
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(ctuple.components())), len(list(ctuple.components(active=None))))
        self.assertEqual(len(list(ctuple.components(active=True))), 0)
        ctuple[-1].activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        ctuple.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for i, c in enumerate(ctuple):
            if i == len(ctuple) - 1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for i, c in enumerate(ctuple.components(active=None)):
            if i == len(ctuple) - 1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(ctuple.components())), len(list(ctuple.components(active=None))))
        self.assertEqual(len(list(ctuple.components(active=True))), 1)
        ctuple.deactivate(shallow=False)
        ctuple.activate(shallow=False)
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for i, c in enumerate(ctuple):
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())), len(list(ctuple.components(active=True))))

    def test_preorder_traversal(self):
        ctuple, traversal = super(_TestActiveTupleContainerBase, self).test_preorder_traversal()
        ctuple[1].deactivate()
        self.assertEqual([None, '[0]', '[2]'], [c.name for c in pmo.preorder_traversal(ctuple, active=True)])
        self.assertEqual([id(ctuple), id(ctuple[0]), id(ctuple[2])], [id(c) for c in pmo.preorder_traversal(ctuple, active=True)])
        ctuple[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active], [c.name for c in pmo.preorder_traversal(ctuple, active=True)])
        self.assertEqual([id(c) for c in traversal if c.active], [id(c) for c in pmo.preorder_traversal(ctuple, active=True)])
        ctuple.deactivate()
        self.assertEqual(len(list(pmo.preorder_traversal(ctuple, active=True))), 0)
        self.assertEqual(len(list(pmo.generate_names(ctuple, active=True))), 0)

    def test_preorder_traversal_descend_check(self):
        ctuple, traversal = super(_TestActiveTupleContainerBase, self).test_preorder_traversal_descend_check()
        ctuple[1].deactivate()

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return True
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=True, descend=descend))
        self.assertEqual([None, '[0]', '[2]'], [c.name for c in order])
        self.assertEqual([id(ctuple), id(ctuple[0]), id(ctuple[2])], [id(c) for c in order])
        if ctuple.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(ctuple), id(ctuple[0]), id(ctuple[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None], [c.name for c in descend.seen])
            self.assertEqual([id(ctuple)], [id(c) for c in descend.seen])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=None, descend=descend))
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
        self.assertEqual([id(ctuple), id(ctuple[0]), id(ctuple[1]), id(ctuple[2])], [id(c) for c in order])
        if ctuple.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(ctuple), id(ctuple[0]), id(ctuple[1]), id(ctuple[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
            self.assertEqual([id(ctuple), id(ctuple[1])], [id(c) for c in descend.seen])
        ctuple[1].deactivate(shallow=False)

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return True
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=True, descend=descend))
        self.assertEqual([c.name for c in traversal if c.active], [c.name for c in order])
        self.assertEqual([id(c) for c in traversal if c.active], [id(c) for c in order])
        self.assertEqual([c.name for c in traversal if c.active and c._is_container], [c.name for c in descend.seen])
        self.assertEqual([id(c) for c in traversal if c.active and c._is_container], [id(c) for c in descend.seen])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=None, descend=descend))
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
        self.assertEqual([id(ctuple), id(ctuple[0]), id(ctuple[1]), id(ctuple[2])], [id(c) for c in order])
        if ctuple.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
            self.assertEqual([id(ctuple), id(ctuple[0]), id(ctuple[1]), id(ctuple[2])], [id(c) for c in descend.seen])
        else:
            self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
            self.assertEqual([id(ctuple), id(ctuple[1])], [id(c) for c in descend.seen])
        ctuple.deactivate()

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return True
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=True, descend=descend))
        self.assertEqual(len(descend.seen), 0)
        self.assertEqual(len(list(pmo.generate_names(ctuple, active=True))), 0)

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=None, descend=descend))
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], ctuple)
        ctuple.deactivate(shallow=False)

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return True
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=True, descend=descend))
        self.assertEqual(len(descend.seen), 0)
        self.assertEqual(len(list(pmo.generate_names(ctuple, active=True))), 0)

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active
        descend.seen = []
        order = list(pmo.preorder_traversal(ctuple, active=None, descend=descend))
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], ctuple)