import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
class TestRenamedNLP(unittest.TestCase):

    def test_rename(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(nlp.primals_names(), expected_names)
        renamed_nlp = RenamedNLP(nlp, {'x[0]': 'y[0]', 'x[1]': 'y[1]', 'x[2]': 'y[2]'})
        expected_names = ['y[0]', 'y[1]', 'y[2]']