import flatbuffers
from flatbuffers.compat import import_numpy
def ConversionOptionsAddSparsityBlockSizes(builder, sparsityBlockSizes):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(sparsityBlockSizes), 0)