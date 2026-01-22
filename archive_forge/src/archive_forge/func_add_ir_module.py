from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def add_ir_module(self, mod):
    self._raise_if_finalized()
    if self._module is not None:
        raise RuntimeError('CUDACodeLibrary only supports one module')
    self._module = mod