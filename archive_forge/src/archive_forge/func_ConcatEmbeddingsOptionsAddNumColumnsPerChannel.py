import flatbuffers
from flatbuffers.compat import import_numpy
def ConcatEmbeddingsOptionsAddNumColumnsPerChannel(builder, numColumnsPerChannel):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(numColumnsPerChannel), 0)