from __future__ import annotations
import abc
from typing import Any, ClassVar, Iterable
import numpy as np
from onnx import TensorProto
from onnx.defs import get_all_schemas_with_history, get_schema, onnx_opset_version
from onnx.helper import make_node, make_tensor_type_proto, np_dtype_to_tensor_dtype
from onnx.numpy_helper import to_array, unpack_int4
from onnx.onnx_pb import AttributeProto, GraphProto, NodeProto, TypeProto
from onnx.reference.custom_element_types import (
def to_array_extended(tensor: TensorProto) -> np.ndarray:
    """Similar to :func:`to_array` but deals with non-numpy types bfloat16,
    float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz, uint4, int4.
    """
    elem_type = tensor.data_type
    if elem_type == TensorProto.BFLOAT16:
        data = tensor.int32_data
        shape = tuple(tensor.dims)
        y = np.empty(shape, dtype=bfloat16).ravel()
        for i, d in enumerate(data):
            y[i] = d
        return y.reshape(shape)
    if elem_type in (TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E4M3FNUZ, TensorProto.FLOAT8E5M2, TensorProto.FLOAT8E5M2FNUZ):
        m = {TensorProto.FLOAT8E4M3FN: float8e4m3fn, TensorProto.FLOAT8E4M3FNUZ: float8e4m3fnuz, TensorProto.FLOAT8E5M2: float8e5m2, TensorProto.FLOAT8E5M2FNUZ: float8e5m2fnuz}
        if tensor.HasField('raw_data'):
            data = tensor.raw_data
        else:
            data = tensor.int32_data
        shape = tuple(tensor.dims)
        y = np.empty(shape, dtype=m[elem_type]).ravel()
        for i, d in enumerate(data):
            y[i] = d
        return y.reshape(shape)
    if elem_type in (TensorProto.UINT4, TensorProto.INT4):
        if tensor.HasField('raw_data'):
            data = tensor.raw_data
        else:
            data = tensor.int32_data
        shape = tuple(tensor.dims)
        m = {TensorProto.INT4: int4, TensorProto.UINT4: uint4}
        dtype = m[elem_type]
        signed = elem_type == TensorProto.INT4
        y = np.empty(len(data), dtype=dtype).ravel()
        for i, d in enumerate(data):
            y[i] = d
        unpacked_data = unpack_int4(y, dims=shape, signed=signed)
        return unpacked_data.astype(dtype)
    return to_array(tensor)