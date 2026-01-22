import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
class TestManualModel(unittest.TestCase):

    def setUp(self):
        opt = Gurobi()
        opt.config.auto_updates.check_for_new_or_removed_params = False
        opt.config.auto_updates.check_for_new_or_removed_vars = False
        opt.config.auto_updates.check_for_new_or_removed_constraints = False
        opt.config.auto_updates.update_parameters = False
        opt.config.auto_updates.update_vars = False
        opt.config.auto_updates.update_constraints = False
        opt.config.auto_updates.update_named_expressions = False
        self.opt = opt

    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2)
        m.c1 = pe.Constraint(expr=m.y >= 2 * m.x + 1)
        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -10)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 10)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], -0.4)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraints([m.c2])
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 2)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        opt.config.load_solutions = False
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)
        opt.remove_constraints([m.c2])
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-06)
        opt.config.solver_options['FeasibilityTol'] = 1e-07
        opt.config.load_solutions = True
        res = opt.solve(m)
        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-07)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)
        m.x.fix(0)
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 0)
        m.x.unfix()
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)
        m.c2 = pe.Constraint(expr=m.y >= m.x ** 2)
        opt.add_constraints([m.c2])
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 1)
        opt.remove_constraints([m.c2])
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        m.z = pe.Var()
        opt.add_variables([m.z])
        self.assertEqual(opt.get_model_attr('NumVars'), 3)
        opt.remove_variables([m.z])
        del m.z
        self.assertEqual(opt.get_model_attr('NumVars'), 2)

    def test_update1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x ** 2 + m.y ** 2)
        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        opt.remove_constraints([m.c1])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)
        opt.add_constraints([m.c1])
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c2 = pe.Constraint(expr=m.x + m.y == 1)
        opt = self.opt
        opt.config.symbolic_solver_labels = True
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)
        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update3(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x ** 2 + m.y ** 2)
        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x ** 2)
        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update4(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x + m.y)
        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x)
        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update5(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)
        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        opt.remove_sos_constraints([m.c1])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)
        opt.add_sos_constraints([m.c1])
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    def test_update6(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)
        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        m.c2 = pe.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        opt.remove_sos_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)