import flatbuffers
from flatbuffers.compat import import_numpy
def TensorMapAddTensorIndex(builder, tensorIndex):
    builder.PrependUint32Slot(1, tensorIndex, 0)