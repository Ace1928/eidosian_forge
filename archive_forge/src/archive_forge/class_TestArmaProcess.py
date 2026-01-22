from statsmodels.compat.pandas import QUARTER_END
import datetime as dt
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import (
class TestArmaProcess:

    def test_empty_coeff(self):
        process = ArmaProcess()
        assert_equal(process.arcoefs, np.array([]))
        assert_equal(process.macoefs, np.array([]))
        process = ArmaProcess([1, -0.8])
        assert_equal(process.arcoefs, np.array([0.8]))
        assert_equal(process.macoefs, np.array([]))
        process = ArmaProcess(ma=[1, -0.8])
        assert_equal(process.arcoefs, np.array([]))
        assert_equal(process.macoefs, np.array([-0.8]))

    def test_from_roots(self):
        ar = [1.8, -0.9]
        ma = [0.3]
        ar.insert(0, -1)
        ma.insert(0, 1)
        ar_p = -1 * np.array(ar)
        ma_p = ma
        process_direct = ArmaProcess(ar_p, ma_p)
        process = ArmaProcess.from_roots(np.array(process_direct.maroots), np.array(process_direct.arroots))
        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)
        process_direct = ArmaProcess(ar=ar_p)
        process = ArmaProcess.from_roots(arroots=np.array(process_direct.arroots))
        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)
        process_direct = ArmaProcess(ma=ma_p)
        process = ArmaProcess.from_roots(maroots=np.array(process_direct.maroots))
        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)
        process_direct = ArmaProcess()
        process = ArmaProcess.from_roots()
        assert_almost_equal(process.arcoefs, process_direct.arcoefs)
        assert_almost_equal(process.macoefs, process_direct.macoefs)
        assert_almost_equal(process.nobs, process_direct.nobs)
        assert_almost_equal(process.maroots, process_direct.maroots)
        assert_almost_equal(process.arroots, process_direct.arroots)
        assert_almost_equal(process.isinvertible, process_direct.isinvertible)
        assert_almost_equal(process.isstationary, process_direct.isstationary)

    def test_from_coeff(self):
        ar = [1.8, -0.9]
        ma = [0.3]
        process = ArmaProcess.from_coeffs(np.array(ar), np.array(ma))
        ar.insert(0, -1)
        ma.insert(0, 1)
        ar_p = -1 * np.array(ar)
        ma_p = ma
        process_direct = ArmaProcess(ar_p, ma_p)
        assert_equal(process.arcoefs, process_direct.arcoefs)
        assert_equal(process.macoefs, process_direct.macoefs)
        assert_equal(process.nobs, process_direct.nobs)
        assert_equal(process.maroots, process_direct.maroots)
        assert_equal(process.arroots, process_direct.arroots)
        assert_equal(process.isinvertible, process_direct.isinvertible)
        assert_equal(process.isstationary, process_direct.isstationary)

    def test_process_multiplication(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        process2 = ArmaProcess.from_coeffs([0.7])
        process3 = process1 * process2
        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([]))
        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        process2 = ArmaProcess.from_coeffs([0.7])
        process3 = process1 * process2
        assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_equal(process3.macoefs, np.array([0.2]))
        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        process2 = process1 * (np.array([1.0, -0.7]), np.array([1.0]))
        assert_equal(process2.arcoefs, np.array([1.6, -0.7 * 0.9]))
        assert_raises(TypeError, process1.__mul__, [3])

    def test_str_repr(self):
        process1 = ArmaProcess.from_coeffs([0.9], [0.2])
        out = process1.__str__()
        print(out)
        assert_(out.find('AR: [1.0, -0.9]') != -1)
        assert_(out.find('MA: [1.0, 0.2]') != -1)
        out = process1.__repr__()
        assert_(out.find('nobs=100') != -1)
        assert_(out.find('at ' + str(hex(id(process1)))) != -1)

    def test_acf(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        acf = process1.acf(10)
        expected = np.array(0.9) ** np.arange(10.0)
        assert_array_almost_equal(acf, expected)
        acf = process1.acf()
        assert_(acf.shape[0] == process1.nobs)

    def test_pacf(self):
        process1 = ArmaProcess.from_coeffs([0.9])
        pacf = process1.pacf(10)
        expected = np.array([1, 0.9] + [0] * 8)
        assert_array_almost_equal(pacf, expected)
        pacf = process1.pacf()
        assert_(pacf.shape[0] == process1.nobs)

    def test_isstationary(self):
        process1 = ArmaProcess.from_coeffs([1.1])
        assert_equal(process1.isstationary, False)
        process1 = ArmaProcess.from_coeffs([1.8, -0.9])
        assert_equal(process1.isstationary, True)
        process1 = ArmaProcess.from_coeffs([1.5, -0.5])
        print(np.abs(process1.arroots))
        assert_equal(process1.isstationary, False)

    def test_arma2ar(self):
        process1 = ArmaProcess.from_coeffs([], [0.8])
        vals = process1.arma2ar(100)
        assert_almost_equal(vals, (-0.8) ** np.arange(100.0))

    def test_invertroots(self):
        process1 = ArmaProcess.from_coeffs([], [2.5])
        process2 = process1.invertroots(True)
        assert_almost_equal(process2.ma, np.array([1.0, 0.4]))
        process1 = ArmaProcess.from_coeffs([], [0.4])
        process2 = process1.invertroots(True)
        assert_almost_equal(process2.ma, np.array([1.0, 0.4]))
        process1 = ArmaProcess.from_coeffs([], [2.5])
        roots, invertable = process1.invertroots(False)
        assert_equal(invertable, False)
        assert_almost_equal(roots, np.array([1, 0.4]))

    def test_generate_sample(self):
        process = ArmaProcess.from_coeffs([0.9])
        np.random.seed(12345)
        sample = process.generate_sample()
        np.random.seed(12345)
        expected = np.random.randn(100)
        for i in range(1, 100):
            expected[i] = 0.9 * expected[i - 1] + expected[i]
        assert_almost_equal(sample, expected)
        process = ArmaProcess.from_coeffs([1.6, -0.9])
        np.random.seed(12345)
        sample = process.generate_sample()
        np.random.seed(12345)
        expected = np.random.randn(100)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 100):
            expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
        assert_almost_equal(sample, expected)
        process = ArmaProcess.from_coeffs([1.6, -0.9])
        np.random.seed(12345)
        sample = process.generate_sample(burnin=100)
        np.random.seed(12345)
        expected = np.random.randn(200)
        expected[1] = 1.6 * expected[0] + expected[1]
        for i in range(2, 200):
            expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
        assert_almost_equal(sample, expected[100:])
        np.random.seed(12345)
        sample = process.generate_sample(nsample=(100, 5))
        assert_equal(sample.shape, (100, 5))

    def test_impulse_response(self):
        process = ArmaProcess.from_coeffs([0.9])
        ir = process.impulse_response(10)
        assert_almost_equal(ir, 0.9 ** np.arange(10))

    def test_periodogram(self):
        process = ArmaProcess()
        pg = process.periodogram()
        assert_almost_equal(pg[0], np.linspace(0, np.pi, 100, False))
        assert_almost_equal(pg[1], np.sqrt(2 / np.pi) / 2 * np.ones(100))