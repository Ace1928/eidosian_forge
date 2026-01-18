from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np
from scipy.optimize import root
def test_tol_norm(self):

    def norm(x):
        return abs(x[0])
    for method in ['excitingmixing', 'diagbroyden', 'linearmixing', 'anderson', 'broyden1', 'broyden2', 'krylov']:
        root(np.zeros_like, np.zeros(2), method=method, options={'tol_norm': norm})