import flatbuffers
from flatbuffers.compat import import_numpy
def SubGraphAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)