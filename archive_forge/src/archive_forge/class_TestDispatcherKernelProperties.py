import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@skip_on_cudasim("CUDA simulator doesn't implement kernel properties")
class TestDispatcherKernelProperties(CUDATestCase):

    def test_get_regs_per_thread_unspecialized(self):

        @cuda.jit
        def pi_sin_array(x, n):
            i = cuda.grid(1)
            if i < n:
                x[i] = 3.14 * math.sin(x[i])
        N = 10
        arr_f32 = np.zeros(N, dtype=np.float32)
        arr_f64 = np.zeros(N, dtype=np.float64)
        pi_sin_array[1, N](arr_f32, N)
        pi_sin_array[1, N](arr_f64, N)
        sig_f32 = void(float32[::1], int64)
        sig_f64 = void(float64[::1], int64)
        regs_per_thread_f32 = pi_sin_array.get_regs_per_thread(sig_f32)
        regs_per_thread_f64 = pi_sin_array.get_regs_per_thread(sig_f64)
        self.assertIsInstance(regs_per_thread_f32, int)
        self.assertIsInstance(regs_per_thread_f64, int)
        self.assertGreater(regs_per_thread_f32, 0)
        self.assertGreater(regs_per_thread_f64, 0)
        regs_per_thread_all = pi_sin_array.get_regs_per_thread()
        self.assertEqual(regs_per_thread_all[sig_f32.args], regs_per_thread_f32)
        self.assertEqual(regs_per_thread_all[sig_f64.args], regs_per_thread_f64)
        if regs_per_thread_f32 == regs_per_thread_f64:
            print('f32 and f64 variant thread usages are equal.')
            print('This may warrant some investigation. Devices:')
            cuda.detect()

    def test_get_regs_per_thread_specialized(self):

        @cuda.jit(void(float32[::1], int64))
        def pi_sin_array(x, n):
            i = cuda.grid(1)
            if i < n:
                x[i] = 3.14 * math.sin(x[i])
        regs_per_thread = pi_sin_array.get_regs_per_thread()
        self.assertIsInstance(regs_per_thread, int)
        self.assertGreater(regs_per_thread, 0)

    def test_get_const_mem_unspecialized(self):

        @cuda.jit
        def const_fmt_string(val, to_print):
            if to_print:
                print(val)
        const_fmt_string[1, 1](1, False)
        const_fmt_string[1, 1](1.0, False)
        sig_i64 = void(int64, boolean)
        sig_f64 = void(float64, boolean)
        const_mem_size_i64 = const_fmt_string.get_const_mem_size(sig_i64)
        const_mem_size_f64 = const_fmt_string.get_const_mem_size(sig_f64)
        self.assertIsInstance(const_mem_size_i64, int)
        self.assertIsInstance(const_mem_size_f64, int)
        self.assertGreaterEqual(const_mem_size_i64, 6)
        self.assertGreaterEqual(const_mem_size_f64, 4)
        const_mem_size_all = const_fmt_string.get_const_mem_size()
        self.assertEqual(const_mem_size_all[sig_i64.args], const_mem_size_i64)
        self.assertEqual(const_mem_size_all[sig_f64.args], const_mem_size_f64)

    def test_get_const_mem_specialized(self):
        arr = np.arange(32, dtype=np.int64)
        sig = void(int64[::1])

        @cuda.jit(sig)
        def const_array_use(x):
            C = cuda.const.array_like(arr)
            i = cuda.grid(1)
            x[i] = C[i]
        const_mem_size = const_array_use.get_const_mem_size(sig)
        self.assertIsInstance(const_mem_size, int)
        self.assertGreaterEqual(const_mem_size, arr.nbytes)

    def test_get_shared_mem_per_block_unspecialized(self):
        N = 10

        @cuda.jit
        def simple_smem(ary):
            sm = cuda.shared.array(N, dtype=ary.dtype)
            for j in range(N):
                sm[j] = j
            for j in range(N):
                ary[j] = sm[j]
        arr_f32 = np.zeros(N, dtype=np.float32)
        arr_f64 = np.zeros(N, dtype=np.float64)
        simple_smem[1, 1](arr_f32)
        simple_smem[1, 1](arr_f64)
        sig_f32 = void(float32[::1])
        sig_f64 = void(float64[::1])
        sh_mem_f32 = simple_smem.get_shared_mem_per_block(sig_f32)
        sh_mem_f64 = simple_smem.get_shared_mem_per_block(sig_f64)
        self.assertIsInstance(sh_mem_f32, int)
        self.assertIsInstance(sh_mem_f64, int)
        self.assertEqual(sh_mem_f32, N * 4)
        self.assertEqual(sh_mem_f64, N * 8)
        sh_mem_f32_all = simple_smem.get_shared_mem_per_block()
        sh_mem_f64_all = simple_smem.get_shared_mem_per_block()
        self.assertEqual(sh_mem_f32_all[sig_f32.args], sh_mem_f32)
        self.assertEqual(sh_mem_f64_all[sig_f64.args], sh_mem_f64)

    def test_get_shared_mem_per_block_specialized(self):

        @cuda.jit(void(float32[::1]))
        def simple_smem(ary):
            sm = cuda.shared.array(100, dtype=float32)
            i = cuda.grid(1)
            if i == 0:
                for j in range(100):
                    sm[j] = j
            cuda.syncthreads()
            ary[i] = sm[i]
        shared_mem_per_block = simple_smem.get_shared_mem_per_block()
        self.assertIsInstance(shared_mem_per_block, int)
        self.assertEqual(shared_mem_per_block, 400)

    def test_get_max_threads_per_block_unspecialized(self):
        N = 10

        @cuda.jit
        def simple_maxthreads(ary):
            i = cuda.grid(1)
            ary[i] = i
        arr_f32 = np.zeros(N, dtype=np.float32)
        simple_maxthreads[1, 1](arr_f32)
        sig_f32 = void(float32[::1])
        max_threads_f32 = simple_maxthreads.get_max_threads_per_block(sig_f32)
        self.assertIsInstance(max_threads_f32, int)
        self.assertGreater(max_threads_f32, 0)
        max_threads_f32_all = simple_maxthreads.get_max_threads_per_block()
        self.assertEqual(max_threads_f32_all[sig_f32.args], max_threads_f32)

    def test_get_local_mem_per_thread_unspecialized(self):
        N = 1000

        @cuda.jit
        def simple_lmem(ary):
            lm = cuda.local.array(N, dtype=ary.dtype)
            for j in range(N):
                lm[j] = j
            for j in range(N):
                ary[j] = lm[j]
        arr_f32 = np.zeros(N, dtype=np.float32)
        arr_f64 = np.zeros(N, dtype=np.float64)
        simple_lmem[1, 1](arr_f32)
        simple_lmem[1, 1](arr_f64)
        sig_f32 = void(float32[::1])
        sig_f64 = void(float64[::1])
        local_mem_f32 = simple_lmem.get_local_mem_per_thread(sig_f32)
        local_mem_f64 = simple_lmem.get_local_mem_per_thread(sig_f64)
        self.assertIsInstance(local_mem_f32, int)
        self.assertIsInstance(local_mem_f64, int)
        self.assertGreaterEqual(local_mem_f32, N * 4)
        self.assertGreaterEqual(local_mem_f64, N * 8)
        local_mem_all = simple_lmem.get_local_mem_per_thread()
        self.assertEqual(local_mem_all[sig_f32.args], local_mem_f32)
        self.assertEqual(local_mem_all[sig_f64.args], local_mem_f64)

    def test_get_local_mem_per_thread_specialized(self):
        N = 1000

        @cuda.jit(void(float32[::1]))
        def simple_lmem(ary):
            lm = cuda.local.array(N, dtype=ary.dtype)
            for j in range(N):
                lm[j] = j
            for j in range(N):
                ary[j] = lm[j]
        local_mem_per_thread = simple_lmem.get_local_mem_per_thread()
        self.assertIsInstance(local_mem_per_thread, int)
        self.assertGreaterEqual(local_mem_per_thread, N * 4)