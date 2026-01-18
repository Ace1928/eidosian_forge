from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def zip_map_test_case(self, attrs, input_type) -> None:
    graph = self._make_graph([('input', TensorProto.FLOAT, ('N', 3))], [make_node('ZipMap', ['input'], ['output'], **attrs, domain='ai.onnx.ml')], [])
    typ = onnx.helper.make_map_type_proto(input_type, onnx.helper.make_tensor_type_proto(TensorProto.FLOAT, ()))
    self._assert_inferred(graph, [onnx.helper.make_value_info('output', onnx.helper.make_sequence_type_proto(typ))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 18)])