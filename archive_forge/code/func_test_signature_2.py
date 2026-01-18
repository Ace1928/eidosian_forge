from numba.np.ufunc.deviceufunc import GUFuncEngine
import unittest
def test_signature_2(self):
    signature = '(m, n), (n, p) -> (m, p)'
    shapes = ((100, 4, 5), (100, 5, 7))
    expects = dict(ishapes=[(4, 5), (5, 7)], oshapes=[(4, 7)], loopdims=(100,), pinned=[False, False])
    template(signature, shapes, expects)