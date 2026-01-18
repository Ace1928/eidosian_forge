from numba.np.ufunc.deviceufunc import GUFuncEngine
import unittest
def test_signature_4(self):
    signature = '(m, n), (n, p) -> (m, p)'
    shapes = ((4, 5), (5, 7))
    expects = dict(ishapes=[(4, 5), (5, 7)], oshapes=[(4, 7)], loopdims=(), pinned=[False, False])
    template(signature, shapes, expects)