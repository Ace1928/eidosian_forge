import functools
import itertools
import logging
from typing import List, Optional
from unittest.mock import patch
import sympy
import torch
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel
class CUTLASSTemplate(CUDATemplate):
    """
    CUTLASSTemplate is a class that provides a template for generating CUTLASS Templates. Used as a baseclass for the
    CUTLASSGemmTemplate, providing functionality that might also be relevant for non-GEMM CUTLASS Kernels.
    """

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice('\n                #include "cute/tensor.hpp"\n                #include "cutlass/cutlass.h"\n                #include "cutlass/numeric_types.h"\n                #include "cutlass/tensor_ref.h"\n                #include "cutlass/util/host_tensor.h"\n                #include "cutlass/util/reference/host/tensor_fill.h"\n                #include "cutlass/util/reference/device/tensor_fill.h"\n                #include "cutlass/util/device_memory.h"\n            ')
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice('\n                using namespace cute;\n                #define CUTLASS_CHECK(status)                                                      \\\n                {                                                                                  \\\n                  cutlass::Status error = status;                                                  \\\n                  if (error != cutlass::Status::kSuccess) {                                        \\\n                    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \\\n                        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \\\n                    throw std::runtime_error(msg);                                                 \\\n                  }                                                                                \\\n                }\n\n                // Used as pass-through functor in EVT just for type casting / rounding\n                template <typename T>\n                struct identity_op {\n                  CUTLASS_HOST_DEVICE\n                  T operator()(T val) const { return val; }\n                };\n\n            ')
        return res

    def cute_int(self, int_str: str, var_name: str) -> str:
        res = ''
        if int_str in {'1', '1L'}:
            res = 'cute::Int<1>{}'
        else:
            res = int_str
        return f'{res} /* {var_name} */'
    _DTYPE_TO_CUTLASS = {torch.float32: 'float', torch.float64: 'double', torch.float16: 'cutlass::half_t', torch.int32: 'int', torch.int8: 'int8_t', torch.uint8: 'uint8_t', torch.bool: 'bool', torch.bfloat16: 'cutlass::bfloat16_t'}

    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f'({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})'