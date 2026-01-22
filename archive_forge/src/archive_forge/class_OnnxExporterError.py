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
class OnnxExporterError(RuntimeError):
    """Raised when an ONNX exporter error occurs.

    This exception is thrown when there's an error during the ONNX export process.
    It encapsulates the :class:`ONNXProgram` object generated until the failure, allowing
    access to the partial export results and associated metadata.
    """
    onnx_program: Final[ONNXProgram]

    def __init__(self, onnx_program: ONNXProgram, message: str):
        """
        Initializes the OnnxExporterError with the given ONNX program and message.

        Args:
            onnx_program (ONNXProgram): The partial results of the ONNX export.
            message (str): The error message to be displayed.
        """
        super().__init__(message)
        self.onnx_program = onnx_program