from itertools import zip_longest
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.kernel as pmo
from pyomo.util.components import iter_component, rename_components
def test_rename_components(self):
    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
    model.z = pyo.Var(bounds=(10, 20))
    model.obj = pyo.Objective(expr=model.z + model.x[1])

    def con_rule(m, i):
        return m.x[i] + m.z == i
    model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
    model.zcon = pyo.Constraint(expr=model.z >= model.x[2])
    model.b = pyo.Block()
    model.b.bx = pyo.Var([1, 2, 3], initialize=42)
    model.b.bz = pyo.Var(initialize=42)
    model.x_ref = pyo.Reference(model.x)
    model.zcon_ref = pyo.Reference(model.zcon)
    model.b.bx_ref = pyo.Reference(model.b.bx[2])
    c_list = list(model.component_objects(ctype=[pyo.Var, pyo.Constraint, pyo.Objective]))
    name_map = rename_components(model=model, component_list=c_list, prefix='scaled_')
    self.assertEqual(name_map[model.scaled_obj], 'obj')
    self.assertEqual(name_map[model.scaled_x], 'x')
    self.assertEqual(name_map[model.scaled_con], 'con')
    self.assertEqual(name_map[model.scaled_zcon], 'zcon')
    self.assertEqual(name_map[model.b.scaled_bz], 'b.bz')
    self.assertEqual(name_map[model.scaled_x_ref], 'x_ref')
    self.assertEqual(name_map[model.scaled_zcon_ref], 'zcon_ref')
    self.assertEqual(name_map[model.b.scaled_bx_ref], 'b.bx_ref')
    self.assertEqual(model.scaled_obj.name, 'scaled_obj')
    self.assertEqual(model.scaled_x.name, 'scaled_x')
    self.assertEqual(model.scaled_con.name, 'scaled_con')
    self.assertEqual(model.scaled_zcon.name, 'scaled_zcon')
    self.assertEqual(model.b.name, 'b')
    self.assertEqual(model.b.scaled_bz.name, 'b.scaled_bz')
    assert hasattr(model, 'scaled_x_ref')
    for i in model.scaled_x_ref:
        assert model.scaled_x_ref[i] is model.scaled_x[i]
    assert hasattr(model, 'scaled_zcon_ref')
    assert model.scaled_zcon_ref[None] is model.scaled_zcon
    assert hasattr(model.b, 'scaled_bx_ref')
    assert model.b.scaled_bx_ref[None] is model.b.scaled_bx[2]