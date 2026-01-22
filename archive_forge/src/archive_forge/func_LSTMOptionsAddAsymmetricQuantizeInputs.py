import flatbuffers
from flatbuffers.compat import import_numpy
def LSTMOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(4, asymmetricQuantizeInputs, 0)