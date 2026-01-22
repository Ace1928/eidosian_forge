import flatbuffers
from flatbuffers.compat import import_numpy
def TensorMapAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)