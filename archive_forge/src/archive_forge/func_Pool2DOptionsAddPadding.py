import flatbuffers
from flatbuffers.compat import import_numpy
def Pool2DOptionsAddPadding(builder, padding):
    builder.PrependInt8Slot(0, padding, 0)