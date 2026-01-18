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
def test_save_model_with_existing_raw_data_should_override(self) -> None:
    model_file_path = self.get_temp_model_filename()
    original_raw_data = self.model.graph.initializer[0].raw_data
    onnx.save_model(self.model, model_file_path, self.serialization_format, save_as_external_data=True, size_threshold=0)
    self.assertTrue(os.path.isfile(model_file_path))
    model = onnx.load_model(model_file_path, self.serialization_format, load_external_data=False)
    initializer_tensor = model.graph.initializer[0]
    initializer_tensor.raw_data = b'dummpy_raw_data'
    load_external_data_for_tensor(initializer_tensor, self.temp_dir)
    self.assertEqual(initializer_tensor.raw_data, original_raw_data)