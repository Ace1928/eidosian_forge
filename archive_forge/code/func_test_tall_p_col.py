import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_tall_p_col(self):
    a, q, r = self.generate('tall')
    for ndel in range(2, 6):
        for col in range(r.shape[1] - ndel):
            q1, r1 = qr_delete(q, r, col, ndel, which='col', overwrite_qr=False)
            a1 = np.delete(a, slice(col, col + ndel), 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)