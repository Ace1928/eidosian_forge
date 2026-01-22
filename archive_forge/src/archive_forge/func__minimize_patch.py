import multiprocessing.pool
import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import scipy.optimize._minimize
import cirq
import cirq_google as cg
from cirq.experiments import random_rotations_between_grid_interaction_layers_circuit
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import (
def _minimize_patch(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, x0_should_be=None):
    assert method == 'nelder-mead'
    np.testing.assert_allclose(x0_should_be, x0)
    return scipy.optimize.OptimizeResult(fun=0, nit=0, nfev=0, status=0, success=True, message='monkeypatched', x=x0.copy(), final_simplex=None)