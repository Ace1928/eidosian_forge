import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def _eq_to_nlp(self, m, nlp, values):
    indices = nlp.get_equality_constraint_indices([m.eq_con_1, m.eq_con_2])
    reordered_values = [None] * 2
    for i, val in zip(indices, values):
        reordered_values[i] = val
    return reordered_values