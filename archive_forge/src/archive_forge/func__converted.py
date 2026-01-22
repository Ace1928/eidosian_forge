import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def _converted(self, graph: GraphProto, initial_version: OperatorSetIdProto, target_version: int) -> ModelProto:
    orig_model = helper.make_model(graph, producer_name='onnx-test', opset_imports=[initial_version])
    converted_model = onnx.version_converter.convert_version(orig_model, target_version)
    checker.check_model(converted_model)
    return converted_model