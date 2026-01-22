import flatbuffers
from flatbuffers.compat import import_numpy
def ModelAddMetadataBuffer(builder, metadataBuffer):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(metadataBuffer), 0)