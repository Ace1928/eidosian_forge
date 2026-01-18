import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.core.plugins.transform.scaling import ScaleModel
def test_scaling_no_solve(self):
    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
    model.z = pyo.Var(bounds=(10, 20), initialize=15)

    def con_rule(m, i):
        if i == 1:
            return m.x[1] + 2 * m.x[2] + 1 * m.x[3] == 8.0
        if i == 2:
            return m.x[1] + 2 * m.x[2] + 2 * m.x[3] == 11.0
        if i == 3:
            return m.x[1] + 3.0 * m.x[2] + 1 * m.x[3] == 10.0
    model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
    model.zcon = pyo.Constraint(expr=model.z >= model.x[2])
    model.x_ref = pyo.Reference(model.x)
    x_scale = 0.5
    obj_scale = 2.0
    z_scale = -10.0
    con_scale1 = 0.5
    con_scale2 = 2.0
    con_scale3 = -5.0
    zcon_scale = -3.0
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.scaling_factor[model.x] = x_scale
    model.scaling_factor[model.z] = z_scale
    model.scaling_factor[model.con[1]] = con_scale1
    model.scaling_factor[model.con[2]] = con_scale2
    model.scaling_factor[model.con[3]] = con_scale3
    model.scaling_factor[model.zcon] = zcon_scale
    model.scaling_factor[model.x_ref] = x_scale * 2
    scaled_model = pyo.TransformationFactory('core.scale_model').create_using(model)
    self.assertAlmostEqual(pyo.value(model.x[1]), pyo.value(scaled_model.scaled_x[1]) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[2]), pyo.value(scaled_model.scaled_x[2]) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[3]), pyo.value(scaled_model.scaled_x[3]) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.z), pyo.value(scaled_model.scaled_z) / z_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[1].lb), pyo.value(scaled_model.scaled_x[1].lb) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[2].lb), pyo.value(scaled_model.scaled_x[2].lb) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[3].lb), pyo.value(scaled_model.scaled_x[3].lb) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.z.lb), pyo.value(scaled_model.scaled_z.ub) / z_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[1].ub), pyo.value(scaled_model.scaled_x[1].ub) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[2].ub), pyo.value(scaled_model.scaled_x[2].ub) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[3].ub), pyo.value(scaled_model.scaled_x[3].ub) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.z.ub), pyo.value(scaled_model.scaled_z.lb) / z_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[1]), pyo.value(scaled_model.scaled_x_ref[1]) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[2]), pyo.value(scaled_model.scaled_x_ref[2]) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[3]), pyo.value(scaled_model.scaled_x_ref[3]) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[1].lb), pyo.value(scaled_model.scaled_x_ref[1].lb) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[2].lb), pyo.value(scaled_model.scaled_x_ref[2].lb) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[3].lb), pyo.value(scaled_model.scaled_x_ref[3].lb) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.z.lb), pyo.value(scaled_model.scaled_z.ub) / z_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[1].ub), pyo.value(scaled_model.scaled_x_ref[1].ub) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[2].ub), pyo.value(scaled_model.scaled_x_ref[2].ub) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.x[3].ub), pyo.value(scaled_model.scaled_x_ref[3].ub) / x_scale, 4)
    self.assertAlmostEqual(pyo.value(model.con[1]), pyo.value(scaled_model.scaled_con[1]) / con_scale1, 4)
    self.assertAlmostEqual(pyo.value(model.con[2]), pyo.value(scaled_model.scaled_con[2]) / con_scale2, 4)
    self.assertAlmostEqual(pyo.value(model.con[3]), pyo.value(scaled_model.scaled_con[3]) / con_scale3, 4)
    self.assertAlmostEqual(pyo.value(model.zcon), pyo.value(scaled_model.scaled_zcon) / zcon_scale, 4)
    scaled_model.scaled_x[1].set_value(1 * x_scale)
    scaled_model.scaled_x[2].set_value(2 * x_scale)
    scaled_model.scaled_x[3].set_value(3 * x_scale)
    scaled_model.scaled_z.set_value(10 * z_scale)
    pyo.TransformationFactory('core.scale_model').propagate_solution(scaled_model, model)
    self.assertAlmostEqual(pyo.value(model.x[1]), 1, 4)
    self.assertAlmostEqual(pyo.value(model.x[2]), 2, 4)
    self.assertAlmostEqual(pyo.value(model.x[3]), 3, 4)
    self.assertAlmostEqual(pyo.value(model.z), 10, 4)
    self.assertAlmostEqual(pyo.value(model.x_ref[1]), 1, 4)
    self.assertAlmostEqual(pyo.value(model.x_ref[2]), 2, 4)
    self.assertAlmostEqual(pyo.value(model.x_ref[3]), 3, 4)
    self.assertAlmostEqual(pyo.value(model.con[1]), 8, 4)
    self.assertAlmostEqual(pyo.value(model.con[2]), 11, 4)
    self.assertAlmostEqual(pyo.value(model.con[3]), 10, 4)
    self.assertAlmostEqual(pyo.value(model.zcon), -8, 4)