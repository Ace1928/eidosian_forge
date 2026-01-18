import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_nvvm_from_llvm(self):
    m = ir.Module('test_nvvm_from_llvm')
    m.triple = 'nvptx64-nvidia-cuda'
    nvvm.add_ir_version(m)
    fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
    kernel = ir.Function(m, fty, name='mycudakernel')
    bldr = ir.IRBuilder(kernel.append_basic_block('entry'))
    bldr.ret_void()
    nvvm.set_cuda_kernel(kernel)
    m.data_layout = NVVM().data_layout
    ptx = nvvm.llvm_to_ptx(str(m)).decode('utf8')
    self.assertTrue('mycudakernel' in ptx)
    self.assertTrue('.address_size 64' in ptx)