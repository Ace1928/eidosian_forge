import flatbuffers
from flatbuffers.compat import import_numpy
def BidirectionalSequenceRNNOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)