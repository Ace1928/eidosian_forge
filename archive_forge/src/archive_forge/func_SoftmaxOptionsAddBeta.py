import flatbuffers
from flatbuffers.compat import import_numpy
def SoftmaxOptionsAddBeta(builder, beta):
    builder.PrependFloat32Slot(0, beta, 0.0)