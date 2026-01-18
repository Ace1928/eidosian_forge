from numba.np.ufunc.deviceufunc import GUFuncEngine
import unittest
def test_signature_6(self):
    signature = '(), () -> ()'
    shapes = ((5,), (5,))
    expects = dict(ishapes=[(), ()], oshapes=[()], loopdims=(5,), pinned=[False, False])
    template(signature, shapes, expects)