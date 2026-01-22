import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)