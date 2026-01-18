import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def test_set_primals_with_list_error(self):
    m, nlp, proj_nlp = self._get_nlps()
    msg = 'only integer scalar arrays can be converted to a scalar index'
    with self.assertRaisesRegex(TypeError, msg):
        proj_nlp.set_primals([1.0, 2.0])