import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_user_extension(self):
    fake_ptr = ctypes.c_void_p(3735928559)
    dtor_invoked = [0]

    def dtor():
        dtor_invoked[0] += 1
    ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr, size=40, finalizer=dtor)
    self.assertEqual(dtor_invoked[0], 0)
    del ptr
    self.assertEqual(dtor_invoked[0], 1)
    ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr, size=40, finalizer=dtor)
    owned = ptr.own()
    del owned
    self.assertEqual(dtor_invoked[0], 1)
    del ptr
    self.assertEqual(dtor_invoked[0], 2)