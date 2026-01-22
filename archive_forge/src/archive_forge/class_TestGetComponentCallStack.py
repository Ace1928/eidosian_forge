import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
class TestGetComponentCallStack(unittest.TestCase):
    get_attribute = IndexedComponent_slice.get_attribute
    get_item = IndexedComponent_slice.get_item

    def assertSameStack(self, stack1, stack2):
        for (call1, arg1), (call2, arg2) in zip(stack1, stack2):
            self.assertIs(call1, call2)
            self.assertEqual(arg1, arg2)

    def assertCorrectStack(self, comp, pred_stack, context=None):
        act_stack = get_component_call_stack(comp, context=None)
        self.assertSameStack(pred_stack, act_stack)

    def model(self):
        m = pyo.ConcreteModel()
        m.s1 = pyo.Set(initialize=[1, 2, 3])
        m.s2 = pyo.Set(initialize=[('a', 1), ('b', 2)])
        m.v0 = pyo.Var()
        m.v1 = pyo.Var(m.s1)
        m.v2 = pyo.Var(m.s2)

        @m.Block()
        def b(b):

            @b.Block(m.s1)
            def b1(b):
                b.v0 = pyo.Var()
                b.v1 = pyo.Var(m.s1)

                @b.Block()
                def b(b):
                    b.v0 = pyo.Var()
                    b.v2 = pyo.Var(m.s2)

                    @b.Block(m.s1, m.s2)
                    def b2(b, i1, i2, i3=None):
                        b.v0 = pyo.Var()
                        b.v1 = pyo.Var(m.s1)

                        @b.Block()
                        def b(b):
                            b.v0 = pyo.Var()
                            b.v2 = pyo.Var(m.s2)
        return m

    def test_no_context(self):
        m = self.model()
        comp = m.v1[1]
        pred_stack = [(self.get_item, 1), (self.get_attribute, 'v1')]
        self.assertCorrectStack(comp, pred_stack)
        comp = m.v1
        pred_stack = [(self.get_attribute, 'v1')]
        self.assertCorrectStack(comp, pred_stack)
        comp = m.b.b1[1].b.b2
        pred_stack = [(self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 1), (self.get_attribute, 'b1'), (self.get_attribute, 'b')]
        self.assertCorrectStack(comp, pred_stack)
        comp = m.b.b1[1].b.b2[1, 'a', 1]
        pred_stack = [(self.get_item, (1, 'a', 1)), (self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 1), (self.get_attribute, 'b1'), (self.get_attribute, 'b')]
        self.assertCorrectStack(comp, pred_stack)
        normalize_index.flatten = False
        comp = m.b.b1[1].b.b2[1, ('a', 1)]
        pred_stack = [(self.get_item, (1, ('a', 1))), (self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 1), (self.get_attribute, 'b1'), (self.get_attribute, 'b')]
        self.assertCorrectStack(comp, pred_stack)
        normalize_index.flatten = True

    def test_from_block(self):
        m = self.model()
        comp = m.v0
        pred_stack = [(self.get_attribute, 'v0')]
        self.assertCorrectStack(comp, pred_stack, context=m)
        comp = m.b.b1[2].b.b2[1, 'a', 1]
        pred_stack = [(self.get_item, (1, 'a', 1)), (self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 2), (self.get_attribute, 'b1'), (self.get_attribute, 'b')]
        self.assertCorrectStack(comp, pred_stack, context=m)
        comp = m.b.b1[2].b.b2[1, 'a', 1].b.v2['b', 2]
        pred_stack = [(self.get_item, ('b', 2)), (self.get_attribute, 'v2'), (self.get_attribute, 'b'), (self.get_item, (1, 'a', 1)), (self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 2)]
        self.assertCorrectStack(comp, pred_stack, context=m.b.b1)
        comp = m.b.b1[2].b.b2
        pred_stack = [(self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 2)]
        self.assertCorrectStack(comp, pred_stack, context=m.b.b1)
        comp = m.b.b1[2]
        pred_stack = [(self.get_item, 2)]
        self.assertCorrectStack(comp, pred_stack, context=m.b.b1)
        comp = m.b.b1
        act_stack = get_component_call_stack(comp, context=m.b.b1)
        self.assertEqual(len(act_stack), 0)

    def test_from_blockdata(self):
        m = self.model()
        context = m.b.b1[3].b.b2[2, 'b', 2]
        comp = m.b.b1[3].b.b2[2, 'b', 2].b
        pred_stack = [(self.get_attribute, 'b')]
        self.assertCorrectStack(comp, pred_stack, context=context)
        comp = m.b.b1[3].b.b2[2, 'b', 2].b.v2['a', 1]
        pred_stack = [(self.get_item, ('a', 1)), (self.get_attribute, 'v2'), (self.get_attribute, 'b')]
        self.assertCorrectStack(comp, pred_stack, context=context)
        context = m.b.b1[3]
        comp = m.b.b1[3]
        act_stack = get_component_call_stack(comp, context=context)
        self.assertEqual(len(act_stack), 0)