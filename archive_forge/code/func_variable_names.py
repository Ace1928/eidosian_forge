from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
@deprecated(msg='This method has been replaced with primals_names', version='6.0.0', remove_in='6.0')
def variable_names(self):
    """Returns ordered list with names of primal variables"""
    return self.primals_names()