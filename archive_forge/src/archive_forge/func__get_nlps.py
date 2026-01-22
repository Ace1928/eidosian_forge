import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def _get_nlps(self):
    m = self._make_model_with_inequalities()
    nlp = PyomoNLP(m)
    primals_ordering = ['x[1]', 'x[0]']
    proj_nlp = ProjectedExtendedNLP(nlp, primals_ordering)
    return (m, nlp, proj_nlp)