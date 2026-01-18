import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.core.plugins.transform.scaling import ScaleModel
def test_scaling_hierarchical(self):
    m = pyo.ConcreteModel()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.v1 = pyo.Var(initialize=10)
    m.v2 = pyo.Var(initialize=20)
    m.v3 = pyo.Var(initialize=30)

    def c1_rule(m):
        return m.v1 == 1000000.0
    m.c1 = pyo.Constraint(rule=c1_rule)

    def c2_rule(m):
        return m.v2 == 0.0001
    m.c2 = pyo.Constraint(rule=c2_rule)
    m.scaling_factor[m.v1] = 1.0
    m.scaling_factor[m.v2] = 0.5
    m.scaling_factor[m.v3] = 0.25
    m.scaling_factor[m.c1] = 1e-05
    m.scaling_factor[m.c2] = 100000.0
    m.b = pyo.Block()
    m.b.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.b.v4 = pyo.Var(initialize=10)
    m.b.v5 = pyo.Var(initialize=20)
    m.b.v6 = pyo.Var(initialize=30)

    def c3_rule(m):
        return m.v4 == 1000000.0
    m.b.c3 = pyo.Constraint(rule=c3_rule)

    def c4_rule(m):
        return m.v5 == 0.0001
    m.b.c4 = pyo.Constraint(rule=c4_rule)
    m.b.scaling_factor[m.b.v4] = 1.0
    m.b.scaling_factor[m.b.v5] = 0.5
    m.b.scaling_factor[m.b.v6] = 0.25
    m.b.scaling_factor[m.b.c3] = 1e-05
    m.b.scaling_factor[m.b.c4] = 100000.0
    values = {}
    values[id(m.v1)] = (m.v1.value, m.scaling_factor[m.v1])
    values[id(m.v2)] = (m.v2.value, m.scaling_factor[m.v2])
    values[id(m.v3)] = (m.v3.value, m.scaling_factor[m.v3])
    values[id(m.c1)] = (pyo.value(m.c1.body), m.scaling_factor[m.c1])
    values[id(m.c2)] = (pyo.value(m.c2.body), m.scaling_factor[m.c2])
    values[id(m.b.v4)] = (m.b.v4.value, m.b.scaling_factor[m.b.v4])
    values[id(m.b.v5)] = (m.b.v5.value, m.b.scaling_factor[m.b.v5])
    values[id(m.b.v6)] = (m.b.v6.value, m.b.scaling_factor[m.b.v6])
    values[id(m.b.c3)] = (pyo.value(m.b.c3.body), m.b.scaling_factor[m.b.c3])
    values[id(m.b.c4)] = (pyo.value(m.b.c4.body), m.b.scaling_factor[m.b.c4])
    m.c2_ref = pyo.Reference(m.c2)
    m.v3_ref = pyo.Reference(m.v3)
    m.b.c4_ref = pyo.Reference(m.b.c4)
    m.b.v6_ref = pyo.Reference(m.b.v6)
    scale = pyo.TransformationFactory('core.scale_model')
    scale.apply_to(m, rename=False)
    self.assertTrue(hasattr(m, 'v1'))
    self.assertTrue(hasattr(m, 'v2'))
    self.assertTrue(hasattr(m, 'c1'))
    self.assertTrue(hasattr(m, 'c2'))
    self.assertTrue(hasattr(m.b, 'v4'))
    self.assertTrue(hasattr(m.b, 'v5'))
    self.assertTrue(hasattr(m.b, 'c3'))
    self.assertTrue(hasattr(m.b, 'c4'))
    orig_val, factor = values[id(m.v1)]
    self.assertAlmostEqual(m.v1.value, orig_val * factor)
    orig_val, factor = values[id(m.v2)]
    self.assertAlmostEqual(m.v2.value, orig_val * factor)
    orig_val, factor = values[id(m.c1)]
    self.assertAlmostEqual(pyo.value(m.c1.body), orig_val * factor)
    orig_val, factor = values[id(m.c2)]
    self.assertAlmostEqual(pyo.value(m.c2.body), orig_val * factor)
    orig_val, factor = values[id(m.v3)]
    self.assertAlmostEqual(m.v3_ref[None].value, orig_val * factor)
    orig_val, factor = values[id(m.b.v4)]
    self.assertAlmostEqual(m.b.v4.value, orig_val * factor)
    orig_val, factor = values[id(m.b.v5)]
    self.assertAlmostEqual(m.b.v5.value, orig_val * factor)
    orig_val, factor = values[id(m.b.c3)]
    self.assertAlmostEqual(pyo.value(m.b.c3.body), orig_val * factor)
    orig_val, factor = values[id(m.b.c4)]
    self.assertAlmostEqual(pyo.value(m.b.c4.body), orig_val * factor)
    orig_val, factor = values[id(m.b.v6)]
    self.assertAlmostEqual(m.b.v6_ref[None].value, orig_val * factor)
    lhs = m.c2.body
    monom_factor = lhs.arg(0)
    scale_factor = m.scaling_factor[m.c2] / m.scaling_factor[m.v2]
    self.assertAlmostEqual(monom_factor, scale_factor)
    lhs = m.b.c4.body
    monom_factor = lhs.arg(0)
    scale_factor = m.b.scaling_factor[m.b.c4] / m.b.scaling_factor[m.b.v5]
    self.assertAlmostEqual(monom_factor, scale_factor)