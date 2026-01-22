import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import tempfile
from pyomo.contrib.pynumero.interfaces.utils import (
class TestAslNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pm = create_pyomo_model1()
        temporary_dir = tempfile.mkdtemp()
        cls.filename = os.path.join(temporary_dir, 'Pyomo_TestAslNLP')
        cls.pm.write(cls.filename + '.nl', io_options={'symbolic_solver_labels': True})

    @classmethod
    def tearDownClass(cls):
        pass

    def test_nlp_interface(self):
        anlp = AslNLP(self.filename)
        execute_extended_nlp_interface(self, anlp)
        self.assertIsNone(anlp.get_obj_scaling())
        self.assertIsNone(anlp.get_primals_scaling())
        self.assertIsNone(anlp.get_constraints_scaling())
        self.assertIsNone(anlp.get_eq_constraints_scaling())
        self.assertIsNone(anlp.get_ineq_constraints_scaling())