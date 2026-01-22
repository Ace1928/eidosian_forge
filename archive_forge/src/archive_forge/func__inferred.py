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
def _inferred(self, graph_or_model: GraphProto | ModelProto, **kwargs: Any) -> ModelProto:
    data_prop = kwargs.pop('data_prop', False)
    if isinstance(graph_or_model, GraphProto):
        kwargs['producer_name'] = 'onnx-test'
        orig_model = helper.make_model(graph_or_model, **kwargs)
    else:
        orig_model = graph_or_model
    inferred_model = onnx.shape_inference.infer_shapes(orig_model, strict_mode=True, data_prop=data_prop)
    checker.check_model(inferred_model)
    return inferred_model