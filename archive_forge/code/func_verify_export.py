from __future__ import annotations
import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import (
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, _exporter_states, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.types import Number
@_beartype.beartype
def verify_export(self, options: VerificationOptions) -> Tuple[Optional[AssertionError], torch.Graph, _OutputsType, _OutputsType]:
    """
        Verify the export from TorchScript IR graph to ONNX.

        Export the TorchScript IR graph to ONNX, with the inputs, parameters and export
        options recorded in this object. Then verify the exported ONNX graph against
        the original TorchScript IR graph under the provided verification options.

        Args:
            options: The verification options.

        Returns:
            error: The AssertionError raised during the verification. Returns None if no
            error is raised.
            onnx_graph: The exported ONNX graph in TorchScript IR format.
            onnx_outs: The outputs from running exported ONNX model under the onnx
            backend in `options`.
            pt_outs: The outputs from running the TorchScript IR graph.
        """
    return verify_aten_graph(self.graph, input_args=self.input_args, params_dict=self.params_dict, export_options=self.export_options, verification_options=options)