import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
class SliceComponentDataAlongSets(unittest.TestCase):

    def model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.space = dae.ContinuousSet(initialize=[0, 2])
        m.comp = pyo.Set(initialize=['a', 'b'])
        m.d_2 = pyo.Set(initialize=[('a', 1), ('b', 2)])
        m.d_none = pyo.Set(initialize=[('c', 1, 10), ('d', 3)], dimen=None)

        @m.Block()
        def b(b):
            b.v0 = pyo.Var()
            b.v1 = pyo.Var(m.time)
            b.v2 = pyo.Var(m.time, m.space)

            @b.Block(m.time, m.space)
            def b2(b2):
                b2.v0 = pyo.Var()
                b2.v1 = pyo.Var(m.comp)
                b2.v2 = pyo.Var(m.time, m.comp)
                b2.vn = pyo.Var(m.time, m.d_none, m.d_2)

            @b.Block(m.d_none, m.d_2)
            def bn(bn):
                bn.v0 = pyo.Var()
                bn.v2 = pyo.Var(m.time, m.space)
                bn.v3 = pyo.Var(m.time, m.space, m.time)
                bn.vn = pyo.Var(m.time, m.d_none, m.d_2)
        return m

    def test_with_tuple_of_sets(self):
        m = pyo.ConcreteModel()
        m.s1 = pyo.Set(initialize=[1, 2, 3])
        m.s2 = pyo.Set(initialize=[1, 2, 3])
        m.v = pyo.Var(m.s1, m.s2)
        sets = (m.s1,)
        slice_ = slice_component_along_sets(m.v[1, 2], sets)
        self.assertEqual(str(pyo.ComponentUID(slice_)), 'v[*,2]')
        self.assertEqual(slice_, m.v[:, 2])

    def test_no_context(self):
        m = self.model()
        comp = m.b.v0
        sets = ComponentSet((m.time, m.space))
        _slice = slice_component_along_sets(comp, sets)
        self.assertIs(_slice, m.b.v0)
        comp = m.b.v1[1]
        sets = ComponentSet((m.time, m.space))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.v1[:])
        comp = m.b.v2[1, 0]
        sets = ComponentSet((m.time, m.space))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.v2[:, :])
        comp = m.b.b2[1, 0].v1['a']
        sets = ComponentSet((m.time, m.space))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.b2[:, :].v1['a'])
        comp = m.b.b2[1, 0].v1
        sets = ComponentSet((m.time, m.space))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.b2[:, :].v1)
        comp = m.b.b2[1, 0].v2[1, 'a']
        sets = ComponentSet((m.time,))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.b2[:, 0].v2[:, 'a'])
        comp = m.b.bn['c', 1, 10, 'a', 1].v3[1, 0, 1]
        sets = ComponentSet((m.time,))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.bn['c', 1, 10, 'a', 1].v3[:, 0, :])

    def test_context(self):
        m = self.model()
        comp = m.b.v2[1, 0]
        context = m.b
        sets = ComponentSet((m.time,))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.v2[:, 0])
        comp = m.b.b2[1, 0].v2[1, 'a']
        context = m.b.b2[1, 0]
        sets = ComponentSet((m.time,))
        _slice = slice_component_along_sets(comp, sets, context=context)
        self.assertEqual(_slice, m.b.b2[1, 0].v2[:, 'a'])
        comp = m.b.b2[1, 0].v1['a']
        context = m.b.b2[1, 0]
        sets = ComponentSet((m.time,))
        _slice = slice_component_along_sets(comp, sets, context=context)
        self.assertIs(_slice, m.b.b2[1, 0].v1['a'])
        sets = ComponentSet((m.comp,))
        _slice = slice_component_along_sets(comp, sets, context=context)
        self.assertEqual(_slice, m.b.b2[1, 0].v1[:])
        context = m.b.b2
        sets = ComponentSet((m.time, m.comp))
        _slice = slice_component_along_sets(comp, sets, context=context)
        self.assertEqual(_slice, m.b.b2[:, 0].v1[:])

    def test_none_dimen(self):
        m = self.model()
        comp = m.b.b2[1, 0].vn
        sets = ComponentSet((m.d_none,))
        _slice = slice_component_along_sets(comp, sets)
        self.assertIs(_slice, m.b.b2[1, 0].vn)
        comp = m.b.b2[1, 0].vn[1, 'd', 3, 'a', 1]
        sets = ComponentSet((m.d_2, m.time))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.b2[:, 0].vn[:, 'd', 3, :, :])
        sets = ComponentSet((m.d_none, m.d_2))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.b2[1, 0].vn[1, ..., :, :])
        comp = m.b.bn['c', 1, 10, 'a', 1].v3[1, 0, 1]
        sets = ComponentSet((m.d_none, m.time))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.bn[..., 'a', 1].v3[:, 0, :])
        comp = m.b.bn['c', 1, 10, 'a', 1].vn[1, 'd', 3, 'b', 2]
        sets = ComponentSet((m.d_none,))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.bn[..., 'a', 1].vn[1, ..., 'b', 2])
        sets = ComponentSet((m.d_none, m.d_2))
        _slice = slice_component_along_sets(comp, sets)
        self.assertEqual(_slice, m.b.bn[..., :, :].vn[1, ..., :, :])
        comp = m.b.bn['c', 1, 10, 'a', 1].vn[1, 'd', 3, 'b', 2]
        context = m.b.bn['c', 1, 10, 'a', 1]
        sets = ComponentSet((m.d_none,))
        _slice = slice_component_along_sets(comp, sets, context=context)
        self.assertEqual(_slice, m.b.bn['c', 1, 10, 'a', 1].vn[1, ..., 'b', 2])
        sets = ComponentSet((m.d_none, m.d_2))
        _slice = slice_component_along_sets(comp, sets, context=context)
        self.assertEqual(_slice, m.b.bn['c', 1, 10, 'a', 1].vn[1, ..., :, :])
        context = m.b.bn
        sets = ComponentSet((m.d_none,))
        _slice = slice_component_along_sets(comp, sets, context=context)
        self.assertEqual(_slice, m.b.bn[..., 'a', 1].vn[1, ..., 'b', 2])