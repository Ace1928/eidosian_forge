from __future__ import annotations
import io
import os
from typing import Tuple, TYPE_CHECKING, Union
import torch
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
@_beartype.beartype
def _create_tensor_proto_with_external_data(tensor: torch.Tensor, name: str, location: str, basepath: str) -> onnx.TensorProto:
    """Create a TensorProto with external data from a PyTorch tensor.
    The external data is saved to os.path.join(basepath, location).

    Args:
        tensor: Tensor to be saved.
        name: Name of the tensor (i.e., initializer name in ONNX graph).
        location: Relative location of the external data file
            (e.g., "/tmp/initializers/weight_0" when model is "/tmp/model_name.onnx").
        basepath: Base path of the external data file (e.g., "/tmp/external_data" while model must be in "/tmp").


    Reference for ONNX's external data format:
        How to load?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L187
        How to save?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L43
        How to set ONNX fields?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L88
    """
    import onnx
    tensor_proto = onnx.TensorProto()
    tensor_proto.name = name
    tensor_proto.data_type = jit_type_utils.JitScalarType.from_dtype(tensor.dtype).onnx_type()
    tensor_proto.dims.extend(tensor.shape)
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL
    key_value_pairs = {'location': location, 'offset': 0, 'length': tensor.untyped_storage().nbytes()}
    for k, v in key_value_pairs.items():
        entry = tensor_proto.external_data.add()
        entry.key = k
        entry.value = str(v)
    external_data_file_path = os.path.join(basepath, location)
    if os.path.exists(external_data_file_path):
        os.remove(external_data_file_path)
    external_data_dir_path = os.path.dirname(external_data_file_path)
    if not os.path.exists(external_data_dir_path):
        os.makedirs(external_data_dir_path)
    with open(external_data_file_path, 'xb') as data_file:
        data_file.write(tensor.numpy(force=True).tobytes())
    return tensor_proto