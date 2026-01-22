import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
class CompilationUnit(object):

    def __init__(self):
        self.driver = NVVM()
        self._handle = nvvm_program()
        err = self.driver.nvvmCreateProgram(byref(self._handle))
        self.driver.check_error(err, 'Failed to create CU')

    def __del__(self):
        driver = NVVM()
        err = driver.nvvmDestroyProgram(byref(self._handle))
        driver.check_error(err, 'Failed to destroy CU', exit=True)

    def add_module(self, buffer):
        """
         Add a module level NVVM IR to a compilation unit.
         - The buffer should contain an NVVM module IR either in the bitcode
           representation (LLVM3.0) or in the text representation.
        """
        err = self.driver.nvvmAddModuleToProgram(self._handle, buffer, len(buffer), None)
        self.driver.check_error(err, 'Failed to add module')

    def lazy_add_module(self, buffer):
        """
        Lazily add an NVVM IR module to a compilation unit.
        The buffer should contain NVVM module IR either in the bitcode
        representation or in the text representation.
        """
        err = self.driver.nvvmLazyAddModuleToProgram(self._handle, buffer, len(buffer), None)
        self.driver.check_error(err, 'Failed to add module')

    def compile(self, **options):
        """Perform Compilation.

        Compilation options are accepted as keyword arguments, with the
        following considerations:

        - Underscores (`_`) in option names are converted to dashes (`-`), to
          match NVVM's option name format.
        - Options that take a value will be emitted in the form
          "-<name>=<value>".
        - Booleans passed as option values will be converted to integers.
        - Options which take no value (such as `-gen-lto`) should have a value
          of `None` passed in and will be emitted in the form "-<name>".

        For documentation on NVVM compilation options, see the CUDA Toolkit
        Documentation:

        https://docs.nvidia.com/cuda/libnvvm-api/index.html#_CPPv418nvvmCompileProgram11nvvmProgramiPPKc
        """

        def stringify_option(k, v):
            k = k.replace('_', '-')
            if v is None:
                return f'-{k}'
            if isinstance(v, bool):
                v = int(v)
            return f'-{k}={v}'
        options = [stringify_option(k, v) for k, v in options.items()]
        c_opts = (c_char_p * len(options))(*[c_char_p(x.encode('utf8')) for x in options])
        err = self.driver.nvvmVerifyProgram(self._handle, len(options), c_opts)
        self._try_error(err, 'Failed to verify\n')
        err = self.driver.nvvmCompileProgram(self._handle, len(options), c_opts)
        self._try_error(err, 'Failed to compile\n')
        reslen = c_size_t()
        err = self.driver.nvvmGetCompiledResultSize(self._handle, byref(reslen))
        self._try_error(err, 'Failed to get size of compiled result.')
        ptxbuf = (c_char * reslen.value)()
        err = self.driver.nvvmGetCompiledResult(self._handle, ptxbuf)
        self._try_error(err, 'Failed to get compiled result.')
        self.log = self.get_log()
        if self.log:
            warnings.warn(self.log, category=NvvmWarning)
        return ptxbuf[:]

    def _try_error(self, err, msg):
        self.driver.check_error(err, '%s\n%s' % (msg, self.get_log()))

    def get_log(self):
        reslen = c_size_t()
        err = self.driver.nvvmGetProgramLogSize(self._handle, byref(reslen))
        self.driver.check_error(err, 'Failed to get compilation log size.')
        if reslen.value > 1:
            logbuf = (c_char * reslen.value)()
            err = self.driver.nvvmGetProgramLog(self._handle, logbuf)
            self.driver.check_error(err, 'Failed to get compilation log.')
            return logbuf.value.decode('utf8')
        return ''