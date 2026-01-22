import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
class TestDLTI:

    def test_dlsim(self):
        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = np.asarray([[0.1, 0.3]])
        d = np.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5
        u = np.hstack((np.linspace(0, 4.0, num=5)[:, np.newaxis], np.full((5, 1), 0.01), np.full((5, 1), -0.002)))
        t_in = np.linspace(0, 2.0, num=5)
        yout_truth = np.array([[-0.001, -0.00073, 0.039446, 0.0915387, 0.13195948]]).T
        xout_truth = np.asarray([[0, 0], [0.0012, 0.0005], [0.40233, 0.00071], [1.163368, -0.079327], [2.2402985, -0.3035679]])
        tout, yout, xout = dlsim((a, b, c, d, dt), u, t_in)
        assert_array_almost_equal(yout_truth, yout)
        assert_array_almost_equal(xout_truth, xout)
        assert_array_almost_equal(t_in, tout)
        dlsim((1, 2, 3), 4)
        u_sparse = u[[0, 4], :]
        t_sparse = np.asarray([0.0, 2.0])
        tout, yout, xout = dlsim((a, b, c, d, dt), u_sparse, t_sparse)
        assert_array_almost_equal(yout_truth, yout)
        assert_array_almost_equal(xout_truth, xout)
        assert_equal(len(tout), yout.shape[0])
        num = np.asarray([1.0, -0.1])
        den = np.asarray([0.3, 1.0, 0.2])
        yout_truth = np.array([[0.0, 0.0, 3.33333333333333, -4.77777777777778, 23.037037037037]]).T
        tout, yout = dlsim((num, den, 0.5), u[:, 0], t_in)
        assert_array_almost_equal(yout, yout_truth)
        assert_array_almost_equal(t_in, tout)
        uflat = np.asarray(u[:, 0])
        uflat = uflat.reshape((5,))
        tout, yout = dlsim((num, den, 0.5), uflat, t_in)
        assert_array_almost_equal(yout, yout_truth)
        assert_array_almost_equal(t_in, tout)
        zd = np.array([0.5, -0.5])
        pd = np.array([1j / np.sqrt(2), -1j / np.sqrt(2)])
        k = 1.0
        yout_truth = np.array([[0.0, 1.0, 2.0, 2.25, 2.5]]).T
        tout, yout = dlsim((zd, pd, k, 0.5), u[:, 0], t_in)
        assert_array_almost_equal(yout, yout_truth)
        assert_array_almost_equal(t_in, tout)
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dlsim, system, u)

    def test_dstep(self):
        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = np.asarray([[0.1, 0.3]])
        d = np.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5
        yout_step_truth = (np.asarray([0.0, 0.04, 0.052, 0.0404, 0.00956, -0.036324, -0.093318, -0.15782348, -0.226628324, -0.2969374948]), np.asarray([-0.1, -0.075, -0.058, -0.04815, -0.04453, -0.0461895, -0.0521812, -0.061588875, -0.073549579, -0.08727047595]), np.asarray([0.0, -0.01, -0.013, -0.0101, -0.00239, 0.009081, 0.0233295, 0.03945587, 0.056657081, 0.0742343737]))
        tout, yout = dstep((a, b, c, d, dt), n=10)
        assert_equal(len(yout), 3)
        for i in range(0, len(yout)):
            assert_equal(yout[i].shape[0], 10)
            assert_array_almost_equal(yout[i].flatten(), yout_step_truth[i])
        tfin = ([1.0], [1.0, 1.0], 0.5)
        yout_tfstep = np.asarray([0.0, 1.0, 0.0])
        tout, yout = dstep(tfin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfstep)
        zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = dstep(zpkin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfstep)
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dstep, system)

    def test_dimpulse(self):
        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = np.asarray([[0.1, 0.3]])
        d = np.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5
        yout_imp_truth = (np.asarray([0.0, 0.04, 0.012, -0.0116, -0.03084, -0.045884, -0.056994, -0.06450548, -0.068804844, -0.0703091708]), np.asarray([-0.1, 0.025, 0.017, 0.00985, 0.00362, -0.0016595, -0.0059917, -0.009407675, -0.011960704, -0.01372089695]), np.asarray([0.0, -0.01, -0.003, 0.0029, 0.00771, 0.011471, 0.0142485, 0.01612637, 0.017201211, 0.0175772927]))
        tout, yout = dimpulse((a, b, c, d, dt), n=10)
        assert_equal(len(yout), 3)
        for i in range(0, len(yout)):
            assert_equal(yout[i].shape[0], 10)
            assert_array_almost_equal(yout[i].flatten(), yout_imp_truth[i])
        tfin = ([1.0], [1.0, 1.0], 0.5)
        yout_tfimpulse = np.asarray([0.0, 1.0, -1.0])
        tout, yout = dimpulse(tfin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)
        zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = dimpulse(zpkin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dimpulse, system)

    def test_dlsim_trivial(self):
        a = np.array([[0.0]])
        b = np.array([[0.0]])
        c = np.array([[0.0]])
        d = np.array([[0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        tout, yout, xout = dlsim((a, b, c, d, 1), u)
        assert_array_equal(tout, np.arange(float(n)))
        assert_array_equal(yout, np.zeros((n, 1)))
        assert_array_equal(xout, np.zeros((n, 1)))

    def test_dlsim_simple1d(self):
        a = np.array([[0.5]])
        b = np.array([[0.0]])
        c = np.array([[1.0]])
        d = np.array([[0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
        assert_array_equal(tout, np.arange(float(n)))
        expected = (0.5 ** np.arange(float(n))).reshape(-1, 1)
        assert_array_equal(yout, expected)
        assert_array_equal(xout, expected)

    def test_dlsim_simple2d(self):
        lambda1 = 0.5
        lambda2 = 0.25
        a = np.array([[lambda1, 0.0], [0.0, lambda2]])
        b = np.array([[0.0], [0.0]])
        c = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[0.0], [0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
        assert_array_equal(tout, np.arange(float(n)))
        expected = np.array([lambda1, lambda2]) ** np.arange(float(n)).reshape(-1, 1)
        assert_array_equal(yout, expected)
        assert_array_equal(xout, expected)

    def test_more_step_and_impulse(self):
        lambda1 = 0.5
        lambda2 = 0.75
        a = np.array([[lambda1, 0.0], [0.0, lambda2]])
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        c = np.array([[1.0, 1.0]])
        d = np.array([[0.0, 0.0]])
        n = 10
        ts, ys = dstep((a, b, c, d, 1), n=n)
        stp0 = 1.0 / (1 - lambda1) * (1.0 - lambda1 ** np.arange(n))
        stp1 = 1.0 / (1 - lambda2) * (1.0 - lambda2 ** np.arange(n))
        assert_allclose(ys[0][:, 0], stp0)
        assert_allclose(ys[1][:, 0], stp1)
        x0 = np.array([1.0, 1.0])
        ti, yi = dimpulse((a, b, c, d, 1), n=n, x0=x0)
        imp = np.array([lambda1, lambda2]) ** np.arange(-1, n + 1).reshape(-1, 1)
        imp[0, :] = 0.0
        y0 = imp[:n, 0] + np.dot(imp[1:n + 1, :], x0)
        y1 = imp[:n, 1] + np.dot(imp[1:n + 1, :], x0)
        assert_allclose(yi[0][:, 0], y0)
        assert_allclose(yi[1][:, 0], y1)
        system = ([1.0], [1.0, -0.5], 0.1)
        t, (y,) = dstep(system, n=3)
        assert_allclose(t, [0, 0.1, 0.2])
        assert_array_equal(y.T, [[0, 1.0, 1.5]])
        t, (y,) = dimpulse(system, n=3)
        assert_allclose(t, [0, 0.1, 0.2])
        assert_array_equal(y.T, [[0, 1, 0.5]])