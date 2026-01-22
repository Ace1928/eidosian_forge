import flatbuffers
from flatbuffers.compat import import_numpy
def SVDFOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(2, asymmetricQuantizeInputs, 0)