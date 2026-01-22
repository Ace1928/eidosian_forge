import flatbuffers
from flatbuffers.compat import import_numpy
def LeakyReluOptionsAddAlpha(builder, alpha):
    builder.PrependFloat32Slot(0, alpha, 0.0)