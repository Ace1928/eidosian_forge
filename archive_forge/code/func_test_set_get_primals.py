import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def test_set_get_primals(self):
    m, nlp, proj_nlp = self._get_nlps()
    primals = proj_nlp.get_primals()
    np.testing.assert_array_equal(primals, [1.1, 1.1])
    nlp.set_primals(self._x_to_nlp(m, nlp, [1.2, 1.3, 1.4, 1.5]))
    proj_primals = proj_nlp.get_primals()
    np.testing.assert_array_equal(primals, [1.3, 1.2])
    proj_nlp.set_primals(np.array([-1.0, -1.1]))
    np.testing.assert_array_equal(proj_nlp.get_primals(), [-1.0, -1.1])
    np.testing.assert_array_equal(nlp.get_primals(), self._x_to_nlp(m, nlp, [-1.1, -1.0, 1.4, 1.5]))