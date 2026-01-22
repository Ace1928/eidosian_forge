import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
@unittest.skipUnless(AmplInterface.available(), 'AmplInterface is not available')
class TestFsolveNLP(unittest.TestCase):

    def test_solve_simple_nlp(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(xtol=1e-09, maxfev=20, tol=1e-08))
        x, info, ier, msg = solver.solve()
        self.assertEqual(ier, 1)
        variables = [m.x[1], m.x[2], m.x[3]]
        predicted_xorder = [0.92846891, -0.22610731, 0.29465397]
        indices = nlp.get_primal_indices(variables)
        nlp_to_x_indices = [None] * len(variables)
        for i, j in enumerate(indices):
            nlp_to_x_indices[j] = i
        predicted_nlporder = [predicted_xorder[i] for i in nlp_to_x_indices]
        self.assertStructuredAlmostEqual(nlp.get_primals().tolist(), predicted_nlporder)

    def test_solve_max_iter(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(xtol=1e-09, maxfev=10))
        x, info, ier, msg = solver.solve()
        self.assertNotEqual(ier, 1)
        self.assertIn('has reached maxfev', msg)

    def test_solve_too_tight_tol(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(xtol=0.001, maxfev=20, tol=1e-08))
        msg = 'does not satisfy the function tolerance'
        with self.assertRaisesRegex(RuntimeError, msg):
            x, info, ier, msg = solver.solve()