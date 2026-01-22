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
class NVVM(object):
    """Process-wide singleton.
    """
    _PROTOTYPES = {'nvvmVersion': (nvvm_result, POINTER(c_int), POINTER(c_int)), 'nvvmCreateProgram': (nvvm_result, POINTER(nvvm_program)), 'nvvmDestroyProgram': (nvvm_result, POINTER(nvvm_program)), 'nvvmAddModuleToProgram': (nvvm_result, nvvm_program, c_char_p, c_size_t, c_char_p), 'nvvmLazyAddModuleToProgram': (nvvm_result, nvvm_program, c_char_p, c_size_t, c_char_p), 'nvvmCompileProgram': (nvvm_result, nvvm_program, c_int, POINTER(c_char_p)), 'nvvmGetCompiledResultSize': (nvvm_result, nvvm_program, POINTER(c_size_t)), 'nvvmGetCompiledResult': (nvvm_result, nvvm_program, c_char_p), 'nvvmGetProgramLogSize': (nvvm_result, nvvm_program, POINTER(c_size_t)), 'nvvmGetProgramLog': (nvvm_result, nvvm_program, c_char_p), 'nvvmIRVersion': (nvvm_result, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)), 'nvvmVerifyProgram': (nvvm_result, nvvm_program, c_int, POINTER(c_char_p))}
    __INSTANCE = None

    def __new__(cls):
        with _nvvm_lock:
            if cls.__INSTANCE is None:
                cls.__INSTANCE = inst = object.__new__(cls)
                try:
                    inst.driver = open_cudalib('nvvm')
                except OSError as e:
                    cls.__INSTANCE = None
                    errmsg = 'libNVVM cannot be found. Do `conda install cudatoolkit`:\n%s'
                    raise NvvmSupportError(errmsg % e)
                for name, proto in inst._PROTOTYPES.items():
                    func = getattr(inst.driver, name)
                    func.restype = proto[0]
                    func.argtypes = proto[1:]
                    setattr(inst, name, func)
        return cls.__INSTANCE

    def __init__(self):
        ir_versions = self.get_ir_version()
        self._majorIR = ir_versions[0]
        self._minorIR = ir_versions[1]
        self._majorDbg = ir_versions[2]
        self._minorDbg = ir_versions[3]
        self._supported_ccs = get_supported_ccs()

    @property
    def data_layout(self):
        if (self._majorIR, self._minorIR) < (1, 8):
            return _datalayout_original
        else:
            return _datalayout_i128

    @property
    def supported_ccs(self):
        return self._supported_ccs

    def get_version(self):
        major = c_int()
        minor = c_int()
        err = self.nvvmVersion(byref(major), byref(minor))
        self.check_error(err, 'Failed to get version.')
        return (major.value, minor.value)

    def get_ir_version(self):
        majorIR = c_int()
        minorIR = c_int()
        majorDbg = c_int()
        minorDbg = c_int()
        err = self.nvvmIRVersion(byref(majorIR), byref(minorIR), byref(majorDbg), byref(minorDbg))
        self.check_error(err, 'Failed to get IR version.')
        return (majorIR.value, minorIR.value, majorDbg.value, minorDbg.value)

    def check_error(self, error, msg, exit=False):
        if error:
            exc = NvvmError(msg, RESULT_CODE_NAMES[error])
            if exit:
                print(exc)
                sys.exit(1)
            else:
                raise exc