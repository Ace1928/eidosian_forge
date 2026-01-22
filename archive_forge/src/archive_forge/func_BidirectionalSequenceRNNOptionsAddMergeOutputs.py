import flatbuffers
from flatbuffers.compat import import_numpy
def BidirectionalSequenceRNNOptionsAddMergeOutputs(builder, mergeOutputs):
    builder.PrependBoolSlot(2, mergeOutputs, 0)