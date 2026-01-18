import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_block_single2(self):
    model = ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 10))

    def _block_rule(_b_, t):
        m = _b_.model()
        b = Block()
        b.s1 = Set(initialize=['A1', 'A2', 'A3'])

        def _init(m, j):
            return j * 2
        b.p1 = Param(m.t, default=_init)
        b.v1 = Var(m.t, initialize=5)
        b.v2 = Var(m.t, initialize=2)
        b.v3 = Var(m.t, b.s1, initialize=1)

        def _con1(_b, ti):
            return _b.v1[ti] * _b.p1[ti] == _b.v1[t] ** 2
        b.con1 = Constraint(m.t, rule=_con1)

        def _con2(_b, i, ti):
            return _b.v2[ti] - _b.v3[ti, i] + _b.p1[ti]
        b.con2 = Expression(b.s1, m.t, rule=_con2)
        return b
    model.blk = Block(model.t, rule=_block_rule)
    expansion_map = ComponentMap()
    self.assertTrue(len(model.blk), 2)
    generate_finite_elements(model.t, 5)
    missing_idx = set(model.blk.index_set()) - set(model.blk._data.keys())
    model.blk._dae_missing_idx = missing_idx
    update_contset_indexed_component(model.blk, expansion_map)
    self.assertEqual(len(model.blk), 6)
    self.assertEqual(len(model.blk[10].con1), 2)
    self.assertEqual(len(model.blk[2].con1), 6)
    self.assertEqual(len(model.blk[10].v2), 2)
    self.assertEqual(model.blk[2].p1[2], 4)
    self.assertEqual(model.blk[8].p1[6], 12)
    self.assertEqual(model.blk[4].con1[4](), 15)
    self.assertEqual(model.blk[6].con1[8](), 55)
    self.assertEqual(model.blk[0].con2['A1', 10](), 21)
    self.assertEqual(model.blk[4].con2['A2', 6](), 13)