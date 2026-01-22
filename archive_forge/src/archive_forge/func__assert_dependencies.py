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
@_beartype.beartype
def _assert_dependencies(export_options: ResolvedExportOptions):
    opset_version = export_options.onnx_registry.opset_version

    def missing_package(package_name: str, exc_info: logging._ExcInfoType):
        message = f'Please install the `{package_name}` package (e.g. `python -m pip install {package_name}`).'
        log.fatal(message, exc_info=exc_info)
        return UnsatisfiedDependencyError(package_name, message)

    def missing_opset(package_name: str):
        message = f'The installed `{package_name}` does not support the specified ONNX opset version {opset_version}. Install a newer `{package_name}` package or specify an older opset version.'
        log.fatal(message)
        return UnsatisfiedDependencyError(package_name, message)
    try:
        import onnx
    except ImportError as e:
        raise missing_package('onnx', e) from e
    if onnx.defs.onnx_opset_version() < opset_version:
        raise missing_opset('onnx')
    try:
        import onnxscript
    except ImportError as e:
        raise missing_package('onnxscript', e) from e
    if not isinstance(onnxscript.onnx_opset.all_opsets['', opset_version], onnxscript.values.Opset):
        raise missing_opset('onnxscript')