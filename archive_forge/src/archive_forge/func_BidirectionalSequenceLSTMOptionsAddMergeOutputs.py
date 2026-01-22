import flatbuffers
from flatbuffers.compat import import_numpy
def BidirectionalSequenceLSTMOptionsAddMergeOutputs(builder, mergeOutputs):
    builder.PrependBoolSlot(3, mergeOutputs, 0)