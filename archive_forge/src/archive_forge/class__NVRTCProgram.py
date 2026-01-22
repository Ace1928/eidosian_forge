import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
class _NVRTCProgram(object):

    def __init__(self, src, name='default_program', headers=(), include_names=(), name_expressions=None, method='ptx'):
        self.ptr = None
        if isinstance(src, bytes):
            src = src.decode('UTF-8')
        if isinstance(name, bytes):
            name = name.decode('UTF-8')
        self.src = src
        self.name = name
        self.ptr = nvrtc.createProgram(src, name, headers, include_names)
        self.name_expressions = name_expressions
        self.method = method

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        if is_shutting_down():
            return
        if self.ptr:
            nvrtc.destroyProgram(self.ptr)

    def compile(self, options=(), log_stream=None):
        try:
            if self.name_expressions:
                for ker in self.name_expressions:
                    nvrtc.addNameExpression(self.ptr, ker)
            nvrtc.compileProgram(self.ptr, options)
            mapping = None
            if self.name_expressions:
                mapping = {}
                for ker in self.name_expressions:
                    mapping[ker] = nvrtc.getLoweredName(self.ptr, ker)
            if log_stream is not None:
                log_stream.write(nvrtc.getProgramLog(self.ptr))
            if self.method == 'cubin':
                return (nvrtc.getCUBIN(self.ptr), mapping)
            elif self.method == 'ptx':
                return (nvrtc.getPTX(self.ptr), mapping)
            else:
                raise RuntimeError('Unknown NVRTC compile method')
        except nvrtc.NVRTCError:
            log = nvrtc.getProgramLog(self.ptr)
            raise CompileException(log, self.src, self.name, options, 'nvrtc' if not runtime.is_hip else 'hiprtc')