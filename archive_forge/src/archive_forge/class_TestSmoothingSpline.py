import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
class TestSmoothingSpline:

    def test_invalid_input(self):
        np.random.seed(1234)
        n = 100
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        y = x ** 2 * np.sin(4 * x) + x ** 3 + np.random.normal(0.0, 1.5, n)
        with assert_raises(ValueError):
            make_smoothing_spline(x, y[1:])
        with assert_raises(ValueError):
            make_smoothing_spline(x[1:], y)
        with assert_raises(ValueError):
            make_smoothing_spline(x.reshape(1, n), y)
        with assert_raises(ValueError):
            make_smoothing_spline(x[::-1], y)
        x_dupl = np.copy(x)
        x_dupl[0] = x_dupl[1]
        with assert_raises(ValueError):
            make_smoothing_spline(x_dupl, y)
        x = np.arange(4)
        y = np.ones(4)
        exception_message = '``x`` and ``y`` length must be at least 5'
        with pytest.raises(ValueError, match=exception_message):
            make_smoothing_spline(x, y)

    def test_compare_with_GCVSPL(self):
        """
        Data is generated in the following way:
        >>> np.random.seed(1234)
        >>> n = 100
        >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
        >>> y = np.sin(x) + np.random.normal(scale=.5, size=n)
        >>> np.savetxt('x.csv', x)
        >>> np.savetxt('y.csv', y)

        We obtain the result of performing the GCV smoothing splines
        package (by Woltring, gcvspl) on the sample data points
        using its version for Octave (https://github.com/srkuberski/gcvspl).
        In order to use this implementation, one should clone the repository
        and open the folder in Octave.
        In Octave, we load up ``x`` and ``y`` (generated from Python code
        above):

        >>> x = csvread('x.csv');
        >>> y = csvread('y.csv');

        Then, in order to access the implementation, we compile gcvspl files in
        Octave:

        >>> mex gcvsplmex.c gcvspl.c
        >>> mex spldermex.c gcvspl.c

        The first function computes the vector of unknowns from the dataset
        (x, y) while the second one evaluates the spline in certain points
        with known vector of coefficients.

        >>> c = gcvsplmex( x, y, 2 );
        >>> y0 = spldermex( x, c, 2, x, 0 );

        If we want to compare the results of the gcvspl code, we can save
        ``y0`` in csv file:

        >>> csvwrite('y0.csv', y0);

        """
        data = np.load(data_file('gcvspl.npz'))
        x = data['x']
        y = data['y']
        y_GCVSPL = data['y_GCVSPL']
        y_compr = make_smoothing_spline(x, y)(x)
        assert_allclose(y_compr, y_GCVSPL, atol=0.0001, rtol=0.0001)

    def test_non_regularized_case(self):
        """
        In case the regularization parameter is 0, the resulting spline
        is an interpolation spline with natural boundary conditions.
        """
        np.random.seed(1234)
        n = 100
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        y = x ** 2 * np.sin(4 * x) + x ** 3 + np.random.normal(0.0, 1.5, n)
        spline_GCV = make_smoothing_spline(x, y, lam=0.0)
        spline_interp = make_interp_spline(x, y, 3, bc_type='natural')
        grid = np.linspace(x[0], x[-1], 2 * n)
        assert_allclose(spline_GCV(grid), spline_interp(grid), atol=1e-15)

    def test_weighted_smoothing_spline(self):
        np.random.seed(1234)
        n = 100
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        y = x ** 2 * np.sin(4 * x) + x ** 3 + np.random.normal(0.0, 1.5, n)
        spl = make_smoothing_spline(x, y)
        for ind in np.random.choice(range(100), size=10):
            w = np.ones(n)
            w[ind] = 30.0
            spl_w = make_smoothing_spline(x, y, w)
            orig = abs(spl(x[ind]) - y[ind])
            weighted = abs(spl_w(x[ind]) - y[ind])
            if orig < weighted:
                raise ValueError(f'Spline with weights should be closer to the points than the original one: {orig:.4} < {weighted:.4}')