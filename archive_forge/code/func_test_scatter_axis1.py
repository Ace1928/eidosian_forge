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
@parameterized.expand(all_versions_for('Scatter'))
def test_scatter_axis1(self, _, version) -> None:
    if version >= 11:
        with self.assertRaises(onnx.checker.ValidationError) as cm:
            self._test_scatter_axis1(version)
        exception = cm.exception
        assert 'Scatter is deprecated' in str(exception)
    else:
        self._test_scatter_axis1(version)