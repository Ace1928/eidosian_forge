from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
@skip_on_cudasim('Simulator does not produce debug dumps')
class TestCudaDebugInfo(CUDATestCase):
    """
    These tests only checks the compiled PTX for debuginfo section
    """

    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        asm = self._getasm(fn, sig=sig)
        re_section_dbginfo = re.compile('\\.section\\s+\\.debug_info\\s+{')
        match = re_section_dbginfo.search(asm)
        assertfn = self.assertIsNotNone if expect else self.assertIsNone
        assertfn(match, msg=asm)

    def test_no_debuginfo_in_asm(self):

        @cuda.jit(debug=False)
        def foo(x):
            x[0] = 1
        self._check(foo, sig=(types.int32[:],), expect=False)

    def test_debuginfo_in_asm(self):

        @cuda.jit(debug=True, opt=False)
        def foo(x):
            x[0] = 1
        self._check(foo, sig=(types.int32[:],), expect=True)

    def test_environment_override(self):
        with override_config('CUDA_DEBUGINFO_DEFAULT', 1):

            @cuda.jit(opt=False)
            def foo(x):
                x[0] = 1
            self._check(foo, sig=(types.int32[:],), expect=True)

            @cuda.jit(debug=False)
            def bar(x):
                x[0] = 1
            self._check(bar, sig=(types.int32[:],), expect=False)

    def test_issue_5835(self):

        @cuda.jit((types.int32[::1],), debug=True, opt=False)
        def f(x):
            x[0] = 0

    def test_wrapper_has_debuginfo(self):
        sig = (types.int32[::1],)

        @cuda.jit(sig, debug=True, opt=0)
        def f(x):
            x[0] = 1
        llvm_ir = f.inspect_llvm(sig)
        defines = [line for line in llvm_ir.splitlines() if 'define void @"_ZN6cudapy' in line]
        self.assertEqual(len(defines), 1)
        wrapper_define = defines[0]
        self.assertIn('!dbg', wrapper_define)

    def test_debug_function_calls_internal_impl(self):

        @cuda.jit((types.int32[:], types.int32[:]), debug=True, opt=False)
        def f(inp, outp):
            outp[0] = 1 if inp[0] in (2, 3) else 3

    def test_debug_function_calls_device_function(self):

        @cuda.jit(device=True, debug=True, opt=0)
        def threadid():
            return cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

        @cuda.jit((types.int32[:],), debug=True, opt=0)
        def kernel(arr):
            i = cuda.grid(1)
            if i < len(arr):
                arr[i] = threadid()

    def _test_chained_device_function(self, kernel_debug, f1_debug, f2_debug):

        @cuda.jit(device=True, debug=f2_debug, opt=False)
        def f2(x):
            return x + 1

        @cuda.jit(device=True, debug=f1_debug, opt=False)
        def f1(x, y):
            return x - f2(y)

        @cuda.jit((types.int32, types.int32), debug=kernel_debug, opt=False)
        def kernel(x, y):
            f1(x, y)
        kernel[1, 1](1, 2)

    def test_chained_device_function(self):
        debug_opts = itertools.product(*[(True, False)] * 3)
        for kernel_debug, f1_debug, f2_debug in debug_opts:
            with self.subTest(kernel_debug=kernel_debug, f1_debug=f1_debug, f2_debug=f2_debug):
                self._test_chained_device_function(kernel_debug, f1_debug, f2_debug)

    def _test_chained_device_function_two_calls(self, kernel_debug, f1_debug, f2_debug):

        @cuda.jit(device=True, debug=f2_debug, opt=False)
        def f2(x):
            return x + 1

        @cuda.jit(device=True, debug=f1_debug, opt=False)
        def f1(x, y):
            return x - f2(y)

        @cuda.jit(debug=kernel_debug, opt=False)
        def kernel(x, y):
            f1(x, y)
            f2(x)
        kernel[1, 1](1, 2)

    def test_chained_device_function_two_calls(self):
        debug_opts = itertools.product(*[(True, False)] * 3)
        for kernel_debug, f1_debug, f2_debug in debug_opts:
            with self.subTest(kernel_debug=kernel_debug, f1_debug=f1_debug, f2_debug=f2_debug):
                self._test_chained_device_function_two_calls(kernel_debug, f1_debug, f2_debug)

    def test_chained_device_three_functions(self):

        def three_device_fns(kernel_debug, leaf_debug):

            @cuda.jit(device=True, debug=leaf_debug, opt=False)
            def f3(x):
                return x * x

            @cuda.jit(device=True)
            def f2(x):
                return f3(x) + 1

            @cuda.jit(device=True)
            def f1(x, y):
                return x - f2(y)

            @cuda.jit(debug=kernel_debug, opt=False)
            def kernel(x, y):
                f1(x, y)
            kernel[1, 1](1, 2)
        three_device_fns(kernel_debug=True, leaf_debug=True)
        three_device_fns(kernel_debug=True, leaf_debug=False)
        three_device_fns(kernel_debug=False, leaf_debug=True)
        three_device_fns(kernel_debug=False, leaf_debug=False)