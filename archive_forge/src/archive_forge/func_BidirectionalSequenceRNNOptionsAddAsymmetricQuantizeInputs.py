import flatbuffers
from flatbuffers.compat import import_numpy
def BidirectionalSequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(3, asymmetricQuantizeInputs, 0)