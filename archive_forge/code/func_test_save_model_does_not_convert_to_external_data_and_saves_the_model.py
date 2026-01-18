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
def test_save_model_does_not_convert_to_external_data_and_saves_the_model(self) -> None:
    model_file_path = self.get_temp_model_filename()
    onnx.save_model(self.model, model_file_path, self.serialization_format, save_as_external_data=False)
    self.assertTrue(os.path.isfile(model_file_path))
    model = onnx.load_model(model_file_path, self.serialization_format)
    initializer_tensor = model.graph.initializer[0]
    self.assertFalse(initializer_tensor.HasField('data_location'))
    attribute_tensor = model.graph.node[0].attribute[0].t
    self.assertFalse(attribute_tensor.HasField('data_location'))