from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def _create_empty_module(self, name):
    ir_module = ir.Module(name)
    ir_module.triple = CUDA_TRIPLE
    ir_module.data_layout = nvvm.NVVM().data_layout
    nvvm.add_ir_version(ir_module)
    return ir_module