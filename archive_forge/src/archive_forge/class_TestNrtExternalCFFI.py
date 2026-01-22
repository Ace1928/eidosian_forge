import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
@skip_unless_cffi
class TestNrtExternalCFFI(EnableNRTStatsMixin, TestCase):
    """Testing the use of externally compiled C code that use NRT
    """

    def setUp(self):
        cpu_target.target_context
        super(TestNrtExternalCFFI, self).setUp()

    def compile_cffi_module(self, name, source, cdef):
        from cffi import FFI
        ffi = FFI()
        ffi.set_source(name, source, include_dirs=[include_path()])
        ffi.cdef(cdef)
        tmpdir = temp_directory('cffi_test_{}'.format(name))
        ffi.compile(tmpdir=tmpdir)
        sys.path.append(tmpdir)
        try:
            mod = import_dynamic(name)
        finally:
            sys.path.remove(tmpdir)
        return (ffi, mod)

    def get_nrt_api_table(self):
        from cffi import FFI
        ffi = FFI()
        nrt_get_api = ffi.cast('void* (*)()', _nrt_python.c_helpers['get_api'])
        table = nrt_get_api()
        return table

    def test_manage_memory(self):
        name = '{}_test_manage_memory'.format(self.__class__.__name__)
        source = '\n#include <stdio.h>\n#include "numba/core/runtime/nrt_external.h"\n\nint status = 0;\n\nvoid my_dtor(void *ptr) {\n    free(ptr);\n    status = 0xdead;\n}\n\nNRT_MemInfo* test_nrt_api(NRT_api_functions *nrt) {\n    void * data = malloc(10);\n    NRT_MemInfo *mi = nrt->manage_memory(data, my_dtor);\n    nrt->acquire(mi);\n    nrt->release(mi);\n    status = 0xa110c;\n    return mi;\n}\n        '
        cdef = '\nvoid* test_nrt_api(void *nrt);\nextern int status;\n        '
        ffi, mod = self.compile_cffi_module(name, source, cdef)
        self.assertEqual(mod.lib.status, 0)
        table = self.get_nrt_api_table()
        out = mod.lib.test_nrt_api(table)
        self.assertEqual(mod.lib.status, 659724)
        mi_addr = int(ffi.cast('size_t', out))
        mi = nrt.MemInfo(mi_addr)
        self.assertEqual(mi.refcount, 1)
        del mi
        self.assertEqual(mod.lib.status, 57005)

    def test_allocate(self):
        name = '{}_test_allocate'.format(self.__class__.__name__)
        source = '\n#include <stdio.h>\n#include "numba/core/runtime/nrt_external.h"\n\nNRT_MemInfo* test_nrt_api(NRT_api_functions *nrt, size_t n) {\n    size_t *data = NULL;\n    NRT_MemInfo *mi = nrt->allocate(n);\n    data = nrt->get_data(mi);\n    data[0] = 0xded;\n    data[1] = 0xabc;\n    data[2] = 0xdef;\n    return mi;\n}\n        '
        cdef = 'void* test_nrt_api(void *nrt, size_t n);'
        ffi, mod = self.compile_cffi_module(name, source, cdef)
        table = self.get_nrt_api_table()
        numbytes = 3 * np.dtype(np.intp).itemsize
        out = mod.lib.test_nrt_api(table, numbytes)
        mi_addr = int(ffi.cast('size_t', out))
        mi = nrt.MemInfo(mi_addr)
        self.assertEqual(mi.refcount, 1)
        buffer = ffi.buffer(ffi.cast('char [{}]'.format(numbytes), mi.data))
        arr = np.ndarray(shape=(3,), dtype=np.intp, buffer=buffer)
        np.testing.assert_equal(arr, [3565, 2748, 3567])

    def test_get_api(self):
        from cffi import FFI

        @njit
        def test_nrt_api():
            return NRT_get_api()
        ffi = FFI()
        expect = int(ffi.cast('size_t', self.get_nrt_api_table()))
        got = test_nrt_api()
        self.assertEqual(expect, got)