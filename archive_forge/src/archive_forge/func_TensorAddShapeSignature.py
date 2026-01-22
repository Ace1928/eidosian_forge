import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddShapeSignature(builder, shapeSignature):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(shapeSignature), 0)