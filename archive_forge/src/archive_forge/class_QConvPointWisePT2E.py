import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
class QConvPointWisePT2E(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [stride, padding, dilation, groups, x_scale, x_zp, o_inv_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, o_inv_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = len(inputs) == 5
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.onednn.qconv2d_pointwise', cpp_kernel='onednn::qconv2d_pointwise')
        self.cpp_kernel_key = 'qconv2d_pointwise'
        self.cpp_op_schema = '\n            at::Tensor(\n                at::Tensor act,\n                double act_scale,\n                int64_t act_zero_point,\n                at::Tensor weight,\n                at::Tensor weight_scales,\n                at::Tensor weight_zero_points,\n                c10::optional<at::Tensor> bias,\n                torch::List<int64_t> stride,\n                torch::List<int64_t> padding,\n                torch::List<int64_t> dilation,\n                int64_t groups,\n                double inv_output_scale,\n                int64_t output_zero_point,\n                c10::optional<c10::ScalarType> output_dtype,\n                c10::string_view attr,\n                torch::List<c10::optional<at::Scalar>> scalars,\n                c10::optional<c10::string_view> algorithm)'

    def codegen(self, wrapper):
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())
        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        w_scale, w_zp = (args[-2], args[-1])
        stride, padding, dilation, groups, x_scale, x_zp, o_inv_scale, o_zp, output_dtype, unary_attr, unary_scalars, unary_algorithm = const_args[-12:]
        codegen_args = (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias, stride, padding, dilation, groups, o_inv_scale, o_zp, output_dtype, unary_attr, unary_scalars, unary_algorithm)
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.codegen_kernel_name(), codegen_args, self.cpp_op_schema, self.cpp_kernel_key)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x: 'TensorBox', x_scale: float, x_zp: int, weight: 'TensorBox', w_scale: 'TensorBox', w_zp: 'TensorBox', bias: 'TensorBox', stride_: List[int], padding_: List[int], dilation_: List[int], groups: int, o_inv_scale: float, output_zero_point: int, output_dtype, unary_attr, unary_scalars, unary_algorithm):
        transposed = False
        output_padding = None
        inputs, constant_args, kernel_layout, _ = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups, transposed, output_padding)
        if bias is None:
            constant_args[1], constant_args[2] = (constant_args[2], constant_args[1])
        else:
            constant_args[0], constant_args[1] = (constant_args[1], constant_args[0])
        w_scale.realize()
        w_zp.realize()
        inputs = inputs + [w_scale, w_zp]
        constant_args = constant_args + [x_scale, x_zp, o_inv_scale, output_zero_point, output_dtype, unary_attr, may_convert_to_optional(unary_scalars), unary_algorithm]
        if output_dtype is not None:
            assert output_dtype in [torch.float32, torch.bfloat16]
            kernel_layout.dtype = output_dtype
        return QConvPointWisePT2E(layout=kernel_layout, inputs=inputs, constant_args=constant_args)