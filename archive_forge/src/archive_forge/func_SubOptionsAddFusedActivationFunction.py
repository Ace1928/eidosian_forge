import flatbuffers
from flatbuffers.compat import import_numpy
def SubOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)