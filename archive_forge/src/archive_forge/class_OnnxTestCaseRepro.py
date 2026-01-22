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
class OnnxTestCaseRepro:

    def __init__(self, repro_dir):
        self.repro_dir = repro_dir
        self.proto, self.inputs, self.outputs = onnx_proto_utils.load_test_case(repro_dir)

    @classmethod
    @_beartype.beartype
    def create_test_case_repro(cls, proto: bytes, inputs, outputs, dir: str, name: Optional[str]=None):
        """Create a repro under "{dir}/test_{name}" for an ONNX test case.

        The test case contains the model and the inputs/outputs data. The directory
        structure is as follows:

        dir
        ├── test_<name>
        │   ├── model.onnx
        │   └── test_data_set_0
        │       ├── input_0.pb
        │       ├── input_1.pb
        │       ├── output_0.pb
        │       └── output_1.pb

        Args:
            proto: ONNX model proto.
            inputs: Inputs to the model.
            outputs: Outputs of the model.
            dir: Directory to save the repro.
            name: Name of the test case. If not specified, a name based on current time
                will be generated.
        Returns:
            Path to the repro.
        """
        if name is None:
            name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        return onnx_proto_utils.export_as_test_case(proto, _to_numpy(inputs), _to_numpy(outputs), name, dir)

    @_beartype.beartype
    def validate(self, options: VerificationOptions):
        """Run the ONNX test case with options.backend, and compare with the expected outputs.

        Args:
            options: Options for validation.

        Raise:
            AssertionError: if outputs from options.backend and expected outputs are not
                equal up to specified precision.
        """
        onnx_session = _onnx_backend_session(io.BytesIO(self.proto), options.backend)
        run_outputs = onnx_session.run(None, self.inputs)
        if hasattr(onnx_session, 'get_outputs'):
            output_names = [o.name for o in onnx_session.get_outputs()]
        elif hasattr(onnx_session, 'output_names'):
            output_names = onnx_session.output_names
        else:
            raise ValueError(f'Unknown onnx session type: {type(onnx_session)}')
        expected_outs = [self.outputs[name] for name in output_names]
        _compare_onnx_pytorch_outputs_in_np(run_outputs, expected_outs, options)