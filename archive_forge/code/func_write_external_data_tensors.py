import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def write_external_data_tensors(model: ModelProto, filepath: str) -> ModelProto:
    """Serializes data for all the tensors which have data location set to TensorProto.External.

    Note: This function also strips basepath information from all tensors' external_data fields.

    Arguments:
        model (ModelProto): Model object which is the source of tensors to serialize.
        filepath: System path to the directory which should be treated as base path for external data.

    Returns:
        ModelProto: The modified model object.
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor) and tensor.HasField('raw_data'):
            save_external_data(tensor, filepath)
            tensor.ClearField('raw_data')
    return model