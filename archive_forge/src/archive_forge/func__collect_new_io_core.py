from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
def _collect_new_io_core(self, original_io, io_names_to_extract):
    original_io_map = self._build_name2obj_dict(original_io)
    original_io_names = set(original_io_map)
    s_io_names_to_extract = set(io_names_to_extract)
    io_names_to_keep = s_io_names_to_extract & original_io_names
    new_io_names_to_add = s_io_names_to_extract - original_io_names
    new_io_tensors = []
    for name in io_names_to_keep:
        new_io_tensors.append(original_io_map[name])
    for name in new_io_names_to_add:
        new_io_tensors.append(self.vimap[name])
    new_io_tensors_map = self._build_name2obj_dict(new_io_tensors)
    return [new_io_tensors_map[name] for name in io_names_to_extract]