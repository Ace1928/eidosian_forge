from numba.np.ufunc.deviceufunc import GUFuncEngine
import unittest
def test_signature_5(self):
    signature = '(a), (a) -> (a)'
    shapes = ((5,), (5,))
    expects = dict(ishapes=[(5,), (5,)], oshapes=[(5,)], loopdims=(), pinned=[False, False])
    template(signature, shapes, expects)