from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
@property
def _function_proto(self):
    func_proto = FunctionProto()
    func_proto.ParseFromString(self._function_body)
    return func_proto