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
def test_save_model_with_external_data_multiple_times(self) -> None:
    onnx.save_model(self.model, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, location=None, size_threshold=1024, convert_attribute=True)
    model_without_loading_external = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
    large_input_tensor = model_without_loading_external.graph.initializer[0]
    self.assertTrue(large_input_tensor.HasField('data_location'))
    np.testing.assert_allclose(to_array(large_input_tensor, self.temp_dir), self.large_data)
    small_shape_tensor = model_without_loading_external.graph.initializer[1]
    self.assertTrue(not small_shape_tensor.HasField('data_location'))
    np.testing.assert_allclose(to_array(small_shape_tensor), self.small_data)
    onnx.save_model(model_without_loading_external, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, location=None, size_threshold=0, convert_attribute=True)
    model_without_loading_external = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
    large_input_tensor = model_without_loading_external.graph.initializer[0]
    self.assertTrue(large_input_tensor.HasField('data_location'))
    np.testing.assert_allclose(to_array(large_input_tensor, self.temp_dir), self.large_data)
    small_shape_tensor = model_without_loading_external.graph.initializer[1]
    self.assertTrue(small_shape_tensor.HasField('data_location'))
    np.testing.assert_allclose(to_array(small_shape_tensor, self.temp_dir), self.small_data)