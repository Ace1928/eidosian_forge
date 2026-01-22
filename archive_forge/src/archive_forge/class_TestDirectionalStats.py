import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestDirectionalStats:

    def test_directional_stats_correctness(self):
        decl = -np.deg2rad(np.array([343.2, 62.0, 36.9, 27.0, 359.0, 5.7, 50.4, 357.6, 44.0]))
        incl = -np.deg2rad(np.array([66.1, 68.7, 70.1, 82.1, 79.5, 73.0, 69.3, 58.8, 51.4]))
        data = np.stack((np.cos(incl) * np.cos(decl), np.cos(incl) * np.sin(decl), np.sin(incl)), axis=1)
        dirstats = stats.directional_stats(data)
        directional_mean = dirstats.mean_direction
        mean_rounded = np.round(directional_mean, 4)
        reference_mean = np.array([0.2984, -0.1346, -0.9449])
        assert_allclose(mean_rounded, reference_mean)

    @pytest.mark.parametrize('angles, ref', [([-np.pi / 2, np.pi / 2], 1.0), ([0, 2 * np.pi], 0.0)])
    def test_directional_stats_2d_special_cases(self, angles, ref):
        if callable(ref):
            ref = ref(angles)
        data = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        res = 1 - stats.directional_stats(data).mean_resultant_length
        assert_allclose(res, ref)

    def test_directional_stats_2d(self):
        rng = np.random.default_rng(314499542280078925880191983383461625100)
        testdata = 2 * np.pi * rng.random((1000,))
        testdata_vector = np.stack((np.cos(testdata), np.sin(testdata)), axis=1)
        dirstats = stats.directional_stats(testdata_vector)
        directional_mean = dirstats.mean_direction
        directional_mean_angle = np.arctan2(directional_mean[1], directional_mean[0])
        directional_mean_angle = directional_mean_angle % (2 * np.pi)
        circmean = stats.circmean(testdata)
        assert_allclose(circmean, directional_mean_angle)
        directional_var = 1 - dirstats.mean_resultant_length
        circular_var = stats.circvar(testdata)
        assert_allclose(directional_var, circular_var)

    def test_directional_mean_higher_dim(self):
        data = np.array([[0.8660254, 0.5, 0.0], [0.8660254, -0.5, 0.0]])
        full_array = np.tile(data, (2, 2, 2, 1))
        expected = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        dirstats = stats.directional_stats(full_array, axis=2)
        assert_allclose(expected, dirstats.mean_direction)

    def test_directional_stats_list_ndarray_input(self):
        data = [[0.8660254, 0.5, 0.0], [0.8660254, -0.5, 0]]
        data_array = np.asarray(data)
        res = stats.directional_stats(data)
        ref = stats.directional_stats(data_array)
        assert_allclose(res.mean_direction, ref.mean_direction)
        assert_allclose(res.mean_resultant_length, res.mean_resultant_length)

    def test_directional_stats_1d_error(self):
        data = np.ones((5,))
        message = 'samples must at least be two-dimensional. Instead samples has shape: (5,)'
        with pytest.raises(ValueError, match=re.escape(message)):
            stats.directional_stats(data)

    def test_directional_stats_normalize(self):
        data = np.array([[0.8660254, 0.5, 0.0], [1.7320508, -1.0, 0.0]])
        res = stats.directional_stats(data, normalize=True)
        normalized_data = data / np.linalg.norm(data, axis=-1, keepdims=True)
        ref = stats.directional_stats(normalized_data, normalize=False)
        assert_allclose(res.mean_direction, ref.mean_direction)
        assert_allclose(res.mean_resultant_length, ref.mean_resultant_length)