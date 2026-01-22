import flatbuffers
from flatbuffers.compat import import_numpy
def TransposeConvOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(3, fusedActivationFunction, 0)