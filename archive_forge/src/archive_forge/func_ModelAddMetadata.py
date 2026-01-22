import flatbuffers
from flatbuffers.compat import import_numpy
def ModelAddMetadata(builder, metadata):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(metadata), 0)