from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def _split_jacobian(self, measurement_subset):
    """
        Split jacobian

        Parameters
        ----------
        measurement_subset: the object of the measurement subsets

        Returns
        -------
        jaco_info: split Jacobian
        """
    jaco_info = {}
    for par in self.parameter_names:
        jaco_info[par] = []
        for name in measurement_subset.variable_names:
            try:
                n_all_measure = self.measurement_variables.index(name)
                jaco_info[par].append(self.all_jacobian_info[par][n_all_measure])
            except:
                raise ValueError('Measurement ', name, ' is not in original measurement set.')
    return jaco_info