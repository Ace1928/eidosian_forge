import flatbuffers
from flatbuffers.compat import import_numpy
def Pool2DOptionsAddFilterHeight(builder, filterHeight):
    builder.PrependInt32Slot(4, filterHeight, 0)