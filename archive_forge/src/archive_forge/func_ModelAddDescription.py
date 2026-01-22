import flatbuffers
from flatbuffers.compat import import_numpy
def ModelAddDescription(builder, description):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(description), 0)