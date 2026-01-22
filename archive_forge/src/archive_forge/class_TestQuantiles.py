import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
class TestQuantiles:
    data = [0.706560797, 0.727229578, 0.990399276, 0.927065621, 0.158953014, 0.887764025, 0.239407086, 0.349638551, 0.972791145, 0.149789972, 0.9369477, 0.132359948, 0.046041972, 0.641675031, 0.945530547, 0.224218684, 0.771450991, 0.820257774, 0.336458052, 0.589113496, 0.509736129, 0.696838829, 0.491323573, 0.622767425, 0.775189248, 0.64146145, 0.1184552, 0.77302945, 0.319280007, 0.752229111, 0.047841438, 0.466295911, 0.583850781, 0.840581845, 0.550086491, 0.466470062, 0.504765074, 0.22685596, 0.362641207, 0.891620942, 0.127898691, 0.490094097, 0.044882048, 0.041441695, 0.317976349, 0.504135618, 0.567353033, 0.434617473, 0.636243375, 0.231803616, 0.230154113, 0.160011327, 0.819464108, 0.854706985, 0.438809221, 0.487427267, 0.78690731, 0.408367937, 0.405534192, 0.25044446, 0.995309248, 0.144389588, 0.739947527, 0.953543606, 0.680051621, 0.388382017, 0.863530727, 0.006514031, 0.118007779, 0.924024803, 0.384236354, 0.893687694, 0.626534881, 0.473051932, 0.750134705, 0.241843555, 0.432947602, 0.689538104, 0.136934797, 0.150206859, 0.474335206, 0.907775349, 0.525869295, 0.189184225, 0.854284286, 0.831089744, 0.251637345, 0.587038213, 0.254475554, 0.237781276, 0.82792862, 0.480283781, 0.594514455, 0.213641488, 0.024194386, 0.536668589, 0.699497811, 0.892804071, 0.093835427, 0.731107772]

    def test_hdquantiles(self):
        data = self.data
        assert_almost_equal(ms.hdquantiles(data, [0.0, 1.0]), [0.006514031, 0.995309248])
        hdq = ms.hdquantiles(data, [0.25, 0.5, 0.75])
        assert_almost_equal(hdq, [0.253210762, 0.512847491, 0.762232442])
        data = np.array(data).reshape(10, 10)
        hdq = ms.hdquantiles(data, [0.25, 0.5, 0.75], axis=0)
        assert_almost_equal(hdq[:, 0], ms.hdquantiles(data[:, 0], [0.25, 0.5, 0.75]))
        assert_almost_equal(hdq[:, -1], ms.hdquantiles(data[:, -1], [0.25, 0.5, 0.75]))
        hdq = ms.hdquantiles(data, [0.25, 0.5, 0.75], axis=0, var=True)
        assert_almost_equal(hdq[..., 0], ms.hdquantiles(data[:, 0], [0.25, 0.5, 0.75], var=True))
        assert_almost_equal(hdq[..., -1], ms.hdquantiles(data[:, -1], [0.25, 0.5, 0.75], var=True))

    def test_hdquantiles_sd(self):
        hd_std_errs = ms.hdquantiles_sd(self.data)
        n = len(self.data)
        jdata = np.broadcast_to(self.data, (n, n))
        jselector = np.logical_not(np.eye(n))
        jdata = jdata[jselector].reshape(n, n - 1)
        jdist = ms.hdquantiles(jdata, axis=1)
        jdist_mean = np.mean(jdist, axis=0)
        jstd = ((n - 1) / n * np.sum((jdist - jdist_mean) ** 2, axis=0)) ** 0.5
        assert_almost_equal(hd_std_errs, jstd)
        assert_almost_equal(hd_std_errs, [0.0379258, 0.0380656, 0.0380013])
        two_data_points = ms.hdquantiles_sd([1, 2])
        assert_almost_equal(two_data_points, [0.5, 0.5, 0.5])

    def test_mquantiles_cimj(self):
        ci_lower, ci_upper = ms.mquantiles_cimj(self.data)
        assert_(ci_lower.size == ci_upper.size == 3)