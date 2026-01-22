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
class ProtobufONNXProgramSerializer:
    """Serializes ONNX graph as Protobuf."""

    @_beartype.beartype
    def serialize(self, onnx_program: ONNXProgram, destination: io.BufferedIOBase) -> None:
        import onnx
        if not isinstance(onnx_program.model_proto, onnx.ModelProto):
            raise ValueError('onnx_program.ModelProto is not an onnx.ModelProto')
        destination.write(onnx_program.model_proto.SerializeToString())