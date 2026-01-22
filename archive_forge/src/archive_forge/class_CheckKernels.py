import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
import pytest
import statsmodels.nonparametric.kernels_asymmetric as kern
class CheckKernels:

    def test_kernels(self, case):
        name, bw = case
        rvs = self.rvs
        x_plot = self.x_plot
        kde = []
        kce = []
        for xi in x_plot:
            kde.append(kern.pdf_kernel_asym(xi, rvs, bw, name))
            kce.append(kern.cdf_kernel_asym(xi, rvs, bw, name))
        kde = np.asarray(kde)
        kce = np.asarray(kce)
        amse = ((kde - self.pdf_dgp) ** 2).mean()
        assert_array_less(amse, self.amse_pdf)
        amse = ((kce - self.cdf_dgp) ** 2).mean()
        assert_array_less(amse, self.amse_cdf)

    def test_kernels_vectorized(self, case):
        name, bw = case
        rvs = self.rvs
        x_plot = self.x_plot
        kde = []
        kce = []
        for xi in x_plot:
            kde.append(kern.pdf_kernel_asym(xi, rvs, bw, name))
            kce.append(kern.cdf_kernel_asym(xi, rvs, bw, name))
        kde = np.asarray(kde)
        kce = np.asarray(kce)
        kde1 = kern.pdf_kernel_asym(x_plot, rvs, bw, name)
        kce1 = kern.cdf_kernel_asym(x_plot, rvs, bw, name)
        assert_allclose(kde1, kde, rtol=1e-12)
        assert_allclose(kce1, kce, rtol=1e-12)

    def test_kernels_weights(self, case):
        name, bw = case
        rvs = self.rvs
        x = self.x_plot
        kde2 = kern.pdf_kernel_asym(x, rvs, bw, name)
        kce2 = kern.cdf_kernel_asym(x, rvs, bw, name)
        n = len(rvs)
        w = np.ones(n) / n
        kde1 = kern.pdf_kernel_asym(x, rvs, bw, name, weights=w)
        kce1 = kern.cdf_kernel_asym(x, rvs, bw, name, weights=w)
        assert_allclose(kde1, kde2, rtol=1e-12)
        assert_allclose(kce1, kce2, rtol=1e-12)
        n = len(rvs)
        w = np.ones(n) / n * 2
        kde1 = kern.pdf_kernel_asym(x, rvs, bw, name, weights=w)
        kce1 = kern.cdf_kernel_asym(x, rvs, bw, name, weights=w)
        assert_allclose(kde1, kde2 * 2, rtol=1e-12)
        assert_allclose(kce1, kce2 * 2, rtol=1e-12)