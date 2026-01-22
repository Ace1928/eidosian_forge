from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
def _op_schema_repr(self) -> str:
    return f'OpSchema(\n    name={self.name!r},\n    domain={self.domain!r},\n    since_version={self.since_version!r},\n    doc={self.doc!r},\n    type_constraints={self.type_constraints!r},\n    inputs={self.inputs!r},\n    outputs={self.outputs!r},\n    attributes={self.attributes!r}\n)'