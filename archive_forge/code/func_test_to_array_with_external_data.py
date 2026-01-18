from __future__ import annotations
import itertools
import os
import pathlib
import tempfile
import unittest
import uuid
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import ModelProto, TensorProto, checker, helper, shape_inference
from onnx.external_data_helper import (
from onnx.numpy_helper import from_array, to_array
def test_to_array_with_external_data(self) -> None:
    onnx.save_model(self.model, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=0)
    model = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
    loaded_large_data = to_array(model.graph.initializer[0], self.temp_dir)
    np.testing.assert_allclose(loaded_large_data, self.large_data)