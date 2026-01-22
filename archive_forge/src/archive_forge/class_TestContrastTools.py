from numpy.testing import assert_equal
import numpy as np
class TestContrastTools:

    def __init__(self):
        self.v1name = ['a0', 'a1', 'a2']
        self.v2name = ['b0', 'b1']
        self.d1 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

    def test_dummy_1d(self):
        x = np.array(['F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M'], dtype='|S1')
        d, labels = (np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]), ['gender_F', 'gender_M'])
        res_d, res_labels = dummy_1d(x, varname='gender')
        assert_equal(res_d, d)
        assert_equal(res_labels, labels)

    def test_contrast_product(self):
        res_cp = contrast_product(self.v1name, self.v2name)
        res_t = [0] * 6
        res_t[0] = ['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1', 'a2_b0', 'a2_b1']
        res_t[1] = np.array([[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 1.0]])
        res_t[2] = ['a1_b0-a0_b0', 'a1_b1-a0_b1', 'a2_b0-a0_b0', 'a2_b1-a0_b1']
        res_t[3] = np.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        res_t[4] = ['a0_b1-a0_b0', 'a1_b1-a1_b0', 'a2_b1-a2_b0']
        for ii in range(5):
            np.testing.assert_equal(res_cp[ii], res_t[ii], err_msg=str(ii))

    def test_dummy_limits(self):
        b, e = dummy_limits(self.d1)
        assert_equal(b, np.array([0, 4, 8]))
        assert_equal(e, np.array([4, 8, 12]))