import flatbuffers
from flatbuffers.compat import import_numpy
def L2NormOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)