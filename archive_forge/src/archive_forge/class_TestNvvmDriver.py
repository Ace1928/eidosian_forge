import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestNvvmDriver(unittest.TestCase):

    def get_nvvmir(self):
        versions = NVVM().get_ir_version()
        data_layout = NVVM().data_layout
        return nvvmir_generic.format(data_layout=data_layout, v=versions)

    def test_nvvm_compile_simple(self):
        nvvmir = self.get_nvvmir()
        ptx = nvvm.llvm_to_ptx(nvvmir).decode('utf8')
        self.assertTrue('simple' in ptx)
        self.assertTrue('ave' in ptx)

    def test_nvvm_compile_nullary_option(self):
        if runtime.get_version() < (11, 5):
            self.skipTest('-gen-lto unavailable in this toolkit version')
        nvvmir = self.get_nvvmir()
        ltoir = nvvm.llvm_to_ptx(nvvmir, opt=3, gen_lto=None, arch='compute_52')
        self.assertEqual(ltoir[:4], b'\xedCN\x7f')

    def test_nvvm_bad_option(self):
        msg = '-made-up-option=2 is an unsupported option'
        with self.assertRaisesRegex(NvvmError, msg):
            nvvm.llvm_to_ptx('', made_up_option=2)

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

    def test_used_list(self):
        m = ir.Module('test_used_list')
        m.triple = 'nvptx64-nvidia-cuda'
        m.data_layout = NVVM().data_layout
        nvvm.add_ir_version(m)
        fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
        kernel = ir.Function(m, fty, name='mycudakernel')
        bldr = ir.IRBuilder(kernel.append_basic_block('entry'))
        bldr.ret_void()
        nvvm.set_cuda_kernel(kernel)
        used_lines = [line for line in str(m).splitlines() if 'llvm.used' in line]
        msg = 'Expected exactly one @"llvm.used" array'
        self.assertEqual(len(used_lines), 1, msg)
        used_line = used_lines[0]
        self.assertIn('mycudakernel', used_line)
        self.assertIn('appending global', used_line)
        self.assertIn('section "llvm.metadata"', used_line)

    def test_nvvm_ir_verify_fail(self):
        m = ir.Module('test_bad_ir')
        m.triple = 'unknown-unknown-unknown'
        m.data_layout = NVVM().data_layout
        nvvm.add_ir_version(m)
        with self.assertRaisesRegex(NvvmError, 'Invalid target triple'):
            nvvm.llvm_to_ptx(str(m))

    def _test_nvvm_support(self, arch):
        compute_xx = 'compute_{0}{1}'.format(*arch)
        nvvmir = self.get_nvvmir()
        ptx = nvvm.llvm_to_ptx(nvvmir, arch=compute_xx, ftz=1, prec_sqrt=0, prec_div=0).decode('utf8')
        self.assertIn('.target sm_{0}{1}'.format(*arch), ptx)
        self.assertIn('simple', ptx)
        self.assertIn('ave', ptx)

    def test_nvvm_support(self):
        """Test supported CC by NVVM
        """
        for arch in nvvm.get_supported_ccs():
            self._test_nvvm_support(arch=arch)

    def test_nvvm_warning(self):
        m = ir.Module('test_nvvm_warning')
        m.triple = 'nvptx64-nvidia-cuda'
        m.data_layout = NVVM().data_layout
        nvvm.add_ir_version(m)
        fty = ir.FunctionType(ir.VoidType(), [])
        kernel = ir.Function(m, fty, name='inlinekernel')
        builder = ir.IRBuilder(kernel.append_basic_block('entry'))
        builder.ret_void()
        nvvm.set_cuda_kernel(kernel)
        kernel.attributes.add('noinline')
        with warnings.catch_warnings(record=True) as w:
            nvvm.llvm_to_ptx(str(m))
        self.assertEqual(len(w), 1)
        self.assertIn('overriding noinline attribute', str(w[0]))