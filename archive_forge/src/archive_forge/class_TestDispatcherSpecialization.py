import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@skip_on_cudasim('Specialization not implemented in the simulator')
class TestDispatcherSpecialization(CUDATestCase):

    def _test_no_double_specialize(self, dispatcher, ty):
        with self.assertRaises(RuntimeError) as e:
            dispatcher.specialize(ty)
        self.assertIn('Dispatcher already specialized', str(e.exception))

    def test_no_double_specialize_sig_same_types(self):

        @cuda.jit('void(float32[::1])')
        def f(x):
            pass
        self._test_no_double_specialize(f, float32[::1])

    def test_no_double_specialize_no_sig_same_types(self):

        @cuda.jit
        def f(x):
            pass
        f_specialized = f.specialize(float32[::1])
        self._test_no_double_specialize(f_specialized, float32[::1])

    def test_no_double_specialize_sig_diff_types(self):

        @cuda.jit('void(int32[::1])')
        def f(x):
            pass
        self._test_no_double_specialize(f, float32[::1])

    def test_no_double_specialize_no_sig_diff_types(self):

        @cuda.jit
        def f(x):
            pass
        f_specialized = f.specialize(int32[::1])
        self._test_no_double_specialize(f_specialized, float32[::1])

    def test_specialize_cache_same(self):

        @cuda.jit
        def f(x):
            pass
        self.assertEqual(len(f.specializations), 0)
        f_float32 = f.specialize(float32[::1])
        self.assertEqual(len(f.specializations), 1)
        f_float32_2 = f.specialize(float32[::1])
        self.assertEqual(len(f.specializations), 1)
        self.assertIs(f_float32, f_float32_2)
        f_int32 = f.specialize(int32[::1])
        self.assertEqual(len(f.specializations), 2)
        self.assertIsNot(f_int32, f_float32)

    def test_specialize_cache_same_with_ordering(self):

        @cuda.jit
        def f(x, y):
            pass
        self.assertEqual(len(f.specializations), 0)
        f_f32a_f32a = f.specialize(float32[:], float32[:])
        self.assertEqual(len(f.specializations), 1)
        f_f32c_f32c = f.specialize(float32[::1], float32[::1])
        self.assertEqual(len(f.specializations), 2)
        self.assertIsNot(f_f32a_f32a, f_f32c_f32c)
        f_f32c_f32c_2 = f.specialize(float32[::1], float32[::1])
        self.assertEqual(len(f.specializations), 2)
        self.assertIs(f_f32c_f32c, f_f32c_f32c_2)