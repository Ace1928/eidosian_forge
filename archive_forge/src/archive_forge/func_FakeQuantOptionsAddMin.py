import flatbuffers
from flatbuffers.compat import import_numpy
def FakeQuantOptionsAddMin(builder, min):
    builder.PrependFloat32Slot(0, min, 0.0)