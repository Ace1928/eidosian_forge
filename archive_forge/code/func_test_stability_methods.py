import numpy as np
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.simulation_smoother import (
from numpy.testing import assert_equal
def test_stability_methods(self):
    model = self.model
    model.stability_method = 0
    model.stability_force_symmetry = True
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)
    model.stability_force_symmetry = False
    assert_equal(model.stability_method, 0)
    model.stability_method = 0
    model.set_stability_method(STABILITY_FORCE_SYMMETRY)
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)
    model.stability_method = 0
    model.set_stability_method(stability_method=True)
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)
    model.stability_method = 0
    model.set_stability_method(stability_force_symmetry=True)
    assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)