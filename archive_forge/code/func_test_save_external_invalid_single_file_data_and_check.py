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
@parameterized.parameterized.expand(itertools.product((True, False), (True, False)))
def test_save_external_invalid_single_file_data_and_check(self, use_absolute_path: bool, use_model_path: bool) -> None:
    model = onnx.load_model(self.model_filename, self.serialization_format)
    model_dir = os.path.join(self.temp_dir, 'save_copy')
    os.mkdir(model_dir)
    traversal_external_data_dir = os.path.join(self.temp_dir, 'invlid_external_data')
    os.mkdir(traversal_external_data_dir)
    if use_absolute_path:
        traversal_external_data_location = os.path.join(traversal_external_data_dir, 'tensors.bin')
    else:
        traversal_external_data_location = '../invlid_external_data/tensors.bin'
    external_data_dir = os.path.join(self.temp_dir, 'external_data')
    os.mkdir(external_data_dir)
    new_model_filepath = os.path.join(model_dir, 'model.onnx')

    def convert_model_to_external_data_no_check(model: ModelProto, location: str):
        for tensor in model.graph.initializer:
            if tensor.HasField('raw_data'):
                set_external_data(tensor, location)
    convert_model_to_external_data_no_check(model, location=traversal_external_data_location)
    onnx.save_model(model, new_model_filepath, self.serialization_format)
    if use_model_path:
        with self.assertRaises(onnx.checker.ValidationError):
            _ = onnx.load_model(new_model_filepath, self.serialization_format)
    else:
        onnx_model = onnx.load_model(new_model_filepath, self.serialization_format, load_external_data=False)
        with self.assertRaises(onnx.checker.ValidationError):
            load_external_data_for_model(onnx_model, external_data_dir)