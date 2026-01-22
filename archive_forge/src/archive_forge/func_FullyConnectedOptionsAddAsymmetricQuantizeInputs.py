import flatbuffers
from flatbuffers.compat import import_numpy
def FullyConnectedOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(3, asymmetricQuantizeInputs, 0)