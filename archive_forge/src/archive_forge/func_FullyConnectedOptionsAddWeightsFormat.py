import flatbuffers
from flatbuffers.compat import import_numpy
def FullyConnectedOptionsAddWeightsFormat(builder, weightsFormat):
    builder.PrependInt8Slot(1, weightsFormat, 0)