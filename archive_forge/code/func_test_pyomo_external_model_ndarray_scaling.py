import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.pyomo_ext_cyipopt import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
def test_pyomo_external_model_ndarray_scaling(self):
    m = pyo.ConcreteModel()
    m.Pin = pyo.Var(initialize=100, bounds=(0, None))
    m.c1 = pyo.Var(initialize=1.0, bounds=(0, None))
    m.c2 = pyo.Var(initialize=1.0, bounds=(0, None))
    m.F = pyo.Var(initialize=10, bounds=(0, None))
    m.P1 = pyo.Var()
    m.P2 = pyo.Var()
    m.F_con = pyo.Constraint(expr=m.F == 10)
    m.Pin_con = pyo.Constraint(expr=m.Pin == 100)
    m.obj = pyo.Objective(expr=(m.P1 - 90) ** 2 + (m.P2 - 40) ** 2)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.obj] = 0.1
    m.scaling_factor[m.Pin] = 2.0
    m.scaling_factor[m.c1] = 3.0
    m.scaling_factor[m.c2] = 4.0
    m.scaling_factor[m.F] = 5.0
    m.scaling_factor[m.P1] = 6.0
    m.scaling_factor[m.P2] = 7.0
    m.scaling_factor[m.F_con] = 8.0
    m.scaling_factor[m.Pin_con] = 9.0
    cyipopt_problem = PyomoExternalCyIpoptProblem(pyomo_model=m, ex_input_output_model=PressureDropModel(), inputs=[m.Pin, m.c1, m.c2, m.F], outputs=[m.P1, m.P2], outputs_eqn_scaling=np.asarray([10.0, 11.0], dtype=np.float64), nl_file_options={'file_determinism': 2})
    options = {'hessian_approximation': 'limited-memory', 'nlp_scaling_method': 'user-scaling', 'output_file': '_cyipopt-pyomo-ext-scaling-ndarray.log', 'file_print_level': 10, 'max_iter': 0}
    solver = CyIpoptSolver(cyipopt_problem, options=options)
    x, info = solver.solve(tee=False)
    with open('_cyipopt-pyomo-ext-scaling-ndarray.log', 'r') as fd:
        solver_trace = fd.read()
    cyipopt_problem.close()
    os.remove('_cyipopt-pyomo-ext-scaling-ndarray.log')
    self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
    self.assertIn('output_file = _cyipopt-pyomo-ext-scaling-ndarray.log', solver_trace)
    self.assertIn('objective scaling factor = 0.1', solver_trace)
    self.assertIn('x scaling provided', solver_trace)
    self.assertIn('c scaling provided', solver_trace)
    self.assertIn('d scaling provided', solver_trace)
    self.assertIn('DenseVector "x scaling vector" with 7 elements:', solver_trace)
    self.assertIn('x scaling vector[    1]= 6.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    2]= 7.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    3]= 5.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    4]= 2.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    5]= 1.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    6]= 3.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    7]= 4.0000000000000000e+00', solver_trace)
    self.assertIn('DenseVector "c scaling vector" with 5 elements:', solver_trace)
    self.assertIn('c scaling vector[    1]= 8.0000000000000000e+00', solver_trace)
    self.assertIn('c scaling vector[    2]= 9.0000000000000000e+00', solver_trace)
    self.assertIn('c scaling vector[    3]= 1.0000000000000000e+00', solver_trace)
    self.assertIn('c scaling vector[    4]= 1.0000000000000000e+01', solver_trace)
    self.assertIn('c scaling vector[    5]= 1.1000000000000000e+01', solver_trace)