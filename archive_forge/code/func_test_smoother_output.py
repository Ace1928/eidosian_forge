import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
def test_smoother_output(self, **kwargs):
    res = self.result
    assert_allclose(res.filtered_joint_probabilities, hamilton_ar2_short_filtered_joint_probabilities)
    desired = hamilton_ar2_short_predicted_joint_probabilities
    if desired.ndim > res.predicted_joint_probabilities.ndim:
        desired = desired.sum(axis=-2)
    assert_allclose(res.predicted_joint_probabilities, desired)
    assert_allclose(res.smoothed_joint_probabilities[..., -1], hamilton_ar2_short_smoothed_joint_probabilities[..., -1])
    assert_allclose(res.smoothed_joint_probabilities[..., -2], hamilton_ar2_short_smoothed_joint_probabilities[..., -2])
    assert_allclose(res.smoothed_joint_probabilities[..., -3], hamilton_ar2_short_smoothed_joint_probabilities[..., -3])
    assert_allclose(res.smoothed_joint_probabilities[..., :-3], hamilton_ar2_short_smoothed_joint_probabilities[..., :-3])