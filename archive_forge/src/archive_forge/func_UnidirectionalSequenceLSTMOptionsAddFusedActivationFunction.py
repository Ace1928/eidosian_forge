import flatbuffers
from flatbuffers.compat import import_numpy
def UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)