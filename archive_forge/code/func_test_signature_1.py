from numba.np.ufunc.deviceufunc import GUFuncEngine
import unittest
def test_signature_1(self):
    signature = '(m, n), (n, p) -> (m, p)'
    shapes = ((100, 4, 5), (1, 5, 7))
    expects = dict(ishapes=[(4, 5), (5, 7)], oshapes=[(4, 7)], loopdims=(100,), pinned=[False, True])
    template(signature, shapes, expects)