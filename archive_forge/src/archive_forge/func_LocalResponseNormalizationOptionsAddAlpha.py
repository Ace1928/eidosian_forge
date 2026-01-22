import flatbuffers
from flatbuffers.compat import import_numpy
def LocalResponseNormalizationOptionsAddAlpha(builder, alpha):
    builder.PrependFloat32Slot(2, alpha, 0.0)