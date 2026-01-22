import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
class GAMSTests(unittest.TestCase):

    @unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
    def test_check_expr_eval_py(self):
        with SolverFactory('gams', solver_io='python') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.e = Expression(expr=log10(m.x) + 5)
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.e)
            self.assertRaises(GamsExceptionExecution, opt.solve, m)

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_check_expr_eval_gms(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.e = Expression(expr=log10(m.x) + 5)
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.e)
            self.assertRaises(ValueError, opt.solve, m)

    @unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
    def test_file_removal_py(self):
        with SolverFactory('gams', solver_io='python') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.x)
            tmpdir = mkdtemp()
            results = opt.solve(m, tmpdir=tmpdir)
            self.assertTrue(os.path.exists(tmpdir))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, '_gams_py_gjo0.gms')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, '_gams_py_gjo0.lst')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, '_gams_py_gdb0.gdx')))
            os.rmdir(tmpdir)
            results = opt.solve(m, tmpdir=tmpdir)
            self.assertFalse(os.path.exists(tmpdir))

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_file_removal_gms(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.x)
            tmpdir = mkdtemp()
            results = opt.solve(m, tmpdir=tmpdir)
            self.assertTrue(os.path.exists(tmpdir))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'model.gms')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'output.lst')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'GAMS_MODEL_p.gdx')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'GAMS_MODEL_s.gdx')))
            os.rmdir(tmpdir)
            results = opt.solve(m, tmpdir=tmpdir)
            self.assertFalse(os.path.exists(tmpdir))

    @unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
    def test_keepfiles_py(self):
        with SolverFactory('gams', solver_io='python') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.x)
            tmpdir = mkdtemp()
            results = opt.solve(m, tmpdir=tmpdir, keepfiles=True)
            self.assertTrue(os.path.exists(tmpdir))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, '_gams_py_gjo0.gms')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, '_gams_py_gjo0.lst')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, '_gams_py_gdb0.gdx')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, '_gams_py_gjo0.pf')))
            shutil.rmtree(tmpdir)

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_keepfiles_gms(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.x)
            tmpdir = mkdtemp()
            results = opt.solve(m, tmpdir=tmpdir, keepfiles=True)
            self.assertTrue(os.path.exists(tmpdir))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'model.gms')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'output.lst')))
            if gdxcc_available:
                self.assertTrue(os.path.exists(os.path.join(tmpdir, 'GAMS_MODEL_p.gdx')))
                self.assertTrue(os.path.exists(os.path.join(tmpdir, 'results_s.gdx')))
            else:
                self.assertTrue(os.path.exists(os.path.join(tmpdir, 'results.dat')))
                self.assertTrue(os.path.exists(os.path.join(tmpdir, 'resultsstat.dat')))
            shutil.rmtree(tmpdir)

    @unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
    def test_fixed_var_sign_py(self):
        with SolverFactory('gams', solver_io='python') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.y = Var()
            m.z = Var()
            m.z.fix(-3)
            m.c1 = Constraint(expr=m.x + m.y - m.z == 0)
            m.c2 = Constraint(expr=m.z + m.y - m.z >= -10000)
            m.c3 = Constraint(expr=-3 * m.z + m.y - m.z >= -10000)
            m.c4 = Constraint(expr=-m.z + m.y - m.z >= -10000)
            m.c5 = Constraint(expr=m.x <= 100)
            m.o = Objective(expr=m.x, sense=maximize)
            results = opt.solve(m)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_fixed_var_sign_gms(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.y = Var()
            m.z = Var()
            m.z.fix(-3)
            m.c1 = Constraint(expr=m.x + m.y - m.z == 0)
            m.c2 = Constraint(expr=m.z + m.y - m.z >= -10000)
            m.c3 = Constraint(expr=-3 * m.z + m.y - m.z >= -10000)
            m.c4 = Constraint(expr=-m.z + m.y - m.z >= -10000)
            m.c5 = Constraint(expr=m.x <= 100)
            m.o = Objective(expr=m.x, sense=maximize)
            results = opt.solve(m)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    @unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
    def test_long_var_py(self):
        with SolverFactory('gams', solver_io='python') as opt:
            m = ConcreteModel()
            x = m.a23456789012345678901234567890123456789012345678901234567890123 = Var()
            y = m.b234567890123456789012345678901234567890123456789012345678901234 = Var()
            z = m.c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 = Var()
            w = m.d01234567890 = Var()
            m.c1 = Constraint(expr=x + y + z + w == 0)
            m.c2 = Constraint(expr=x >= 10)
            m.o = Objective(expr=x)
            results = opt.solve(m)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_long_var_gms(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            m = ConcreteModel()
            x = m.a23456789012345678901234567890123456789012345678901234567890123 = Var()
            y = m.b234567890123456789012345678901234567890123456789012345678901234 = Var()
            z = m.c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 = Var()
            w = m.d01234567890 = Var()
            m.c1 = Constraint(expr=x + y + z + w == 0)
            m.c2 = Constraint(expr=x >= 10)
            m.o = Objective(expr=x)
            results = opt.solve(m)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    def test_subsolver_notation(self):
        opt1 = SolverFactory('gams:ipopt', solver_io='gms')
        self.assertTrue(isinstance(opt1, GAMSShell))
        self.assertEqual(opt1.options['solver'], 'ipopt')
        opt2 = SolverFactory('gams:baron', solver_io='python')
        self.assertTrue(isinstance(opt2, GAMSDirect))
        self.assertEqual(opt2.options['solver'], 'baron')
        opt3 = SolverFactory('py:gams')
        self.assertTrue(isinstance(opt3, GAMSDirect))
        opt3.options['keepfiles'] = True
        self.assertEqual(opt3.options['keepfiles'], True)
        opt4 = SolverFactory('py:gams:cbc')
        self.assertTrue(isinstance(opt4, GAMSDirect))
        self.assertEqual(opt4.options['solver'], 'cbc')

    @unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
    def test_options_py(self):
        with SolverFactory('gams', solver_io='python') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.x)
            opt.options['load_solutions'] = False
            opt.solve(m)
            self.assertEqual(m.x.value, None)
            opt.solve(m, load_solutions=True)
            self.assertEqual(m.x.value, 10)

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_options_gms(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=m.x >= 10)
            m.o = Objective(expr=m.x)
            opt.options['load_solutions'] = False
            opt.solve(m)
            self.assertEqual(m.x.value, None)
            opt.solve(m, load_solutions=True)
            self.assertEqual(m.x.value, 10)

    @unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
    def test_version_py(self):
        with SolverFactory('gams', solver_io='python') as opt:
            self.assertIsNotNone(opt.version())

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_version_gms(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            self.assertIsNotNone(opt.version())

    @unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
    def test_dat_parser(self):
        m = pyo.ConcreteModel()
        m.S = pyo.Set(initialize=list(range(5)))
        m.a_long_var_name = pyo.Var(m.S, bounds=(0, 1), initialize=1)
        m.obj = pyo.Objective(expr=2000 * pyo.summation(m.a_long_var_name), sense=pyo.maximize)
        solver = pyo.SolverFactory('gams:conopt')
        res = solver.solve(m, symbolic_solver_labels=True, load_solutions=False, io_options={'put_results_format': 'dat'})
        self.assertEqual(res.solution[0].Objective['obj']['Value'], 10000)
        for i in range(5):
            self.assertEqual(res.solution[0].Variable[f'a_long_var_name_{i}_']['Value'], 1)