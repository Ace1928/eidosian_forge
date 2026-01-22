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
def _assert_inferred(self, graph_or_model: GraphProto | ModelProto, vis: list[ValueInfoProto], **kwargs: Any) -> None:
    graph = graph_or_model if isinstance(graph_or_model, GraphProto) else graph_or_model.graph
    names_in_vis = {x.name for x in vis}
    vis = [x for x in graph.value_info if x.name not in names_in_vis] + vis
    inferred_model = self._inferred(graph_or_model, **kwargs)
    inferred_vis = list(inferred_model.graph.value_info)
    vis = sorted(vis, key=lambda x: x.name)
    inferred_vis = sorted(inferred_vis, key=lambda x: x.name)
    assert len(vis) == len(inferred_vis)
    for v, inferred_v in zip(vis, inferred_vis):
        self._compare_value_infos(v.type, inferred_v.type)