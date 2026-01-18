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
@parameterized.expand(all_versions_for('RegexFullMatch'))
def test_regex_full_match_empty_shape(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.STRING, ())], [make_node('RegexFullMatch', ['x'], ['y'], pattern='^[A-Z][a-z]*$')], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.BOOL, ())], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])