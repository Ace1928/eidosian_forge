import flatbuffers
from flatbuffers.compat import import_numpy
def DivOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)