import flatbuffers
from flatbuffers.compat import import_numpy
def LocalResponseNormalizationOptionsAddBeta(builder, beta):
    builder.PrependFloat32Slot(3, beta, 0.0)