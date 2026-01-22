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
def _build_schemas() -> dict[str, type]:
    res: dict[str, type] = {}
    for schema in get_all_schemas_with_history():
        if schema.name in res:
            if schema.domain != res[schema.name].domain:
                raise NotImplementedError(f'This function assumes every operator has a unique name {schema.name!r} even accross multiple domains {schema.domain!r} and {res[schema.name].domain!r}.')
            if schema.since_version > res[schema.name].since_version:
                res[schema.name] = schema
        else:
            res[schema.name] = schema
        res[schema.name + '_' + str(schema.since_version)] = schema
    return res