import numpy as np
from scipy.optimize import _lbfgsb, minimize
def test_setulb_floatround():
    """test if setulb() violates bounds

    checks for violation due to floating point rounding error
    """
    n = 5
    m = 10
    factr = 10000000.0
    pgtol = 1e-05
    maxls = 20
    iprint = -1
    nbd = np.full((n,), 2)
    low_bnd = np.zeros(n, np.float64)
    upper_bnd = np.ones(n, np.float64)
    x0 = np.array([0.8750000000000278, 0.7500000000000153, 0.9499999999999722, 0.8214285714285992, 0.6363636363636085])
    x = np.copy(x0)
    f = np.array(0.0, np.float64)
    g = np.zeros(n, np.float64)
    fortran_int = _lbfgsb.types.intvar.dtype
    wa = np.zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, np.float64)
    iwa = np.zeros(3 * n, fortran_int)
    task = np.zeros(1, 'S60')
    csave = np.zeros(1, 'S60')
    lsave = np.zeros(4, fortran_int)
    isave = np.zeros(44, fortran_int)
    dsave = np.zeros(29, np.float64)
    task[:] = b'START'
    for n_iter in range(7):
        f, g = objfun(x)
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa, iwa, task, iprint, csave, lsave, isave, dsave, maxls)
        assert (x <= upper_bnd).all() and (x >= low_bnd).all(), '_lbfgsb.setulb() stepped to a point outside of the bounds'