from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
def _initiate_registry_from_torchlib(self, torchlib_registry: torchlib_registry.Registry):
    """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
    for aten_name, aten_overloads_func in torchlib_registry.items():
        internal_name_instance = registration.OpName.from_qualified_name(aten_name)
        for overload_func in aten_overloads_func.overloads:
            symbolic_function = registration.ONNXFunction(onnx_function=overload_func, op_full_name=internal_name_instance.qualified_name(), is_custom=False, is_complex=False)
            self._register(internal_name_instance, symbolic_function)
        for complex_func in aten_overloads_func.complex:
            symbolic_function = registration.ONNXFunction(onnx_function=complex_func, op_full_name=internal_name_instance.qualified_name(), is_custom=False, is_complex=True)
            self._register(internal_name_instance, symbolic_function)