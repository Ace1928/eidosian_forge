import flatbuffers
from flatbuffers.compat import import_numpy
def RNNOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(1, asymmetricQuantizeInputs, 0)