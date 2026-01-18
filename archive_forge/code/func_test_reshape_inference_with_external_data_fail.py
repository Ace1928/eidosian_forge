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
def test_reshape_inference_with_external_data_fail(self) -> None:
    onnx.save_model(self.model, self.model_file_path, self.serialization_format, save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=0)
    model_without_external_data = onnx.load(self.model_file_path, self.serialization_format, load_external_data=False)
    self.assertRaises(shape_inference.InferenceError, shape_inference.infer_shapes, model_without_external_data, strict_mode=True)