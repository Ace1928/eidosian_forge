import flatbuffers
from flatbuffers.compat import import_numpy
def ResizeNearestNeighborOptionsAddAlignCorners(builder, alignCorners):
    builder.PrependBoolSlot(0, alignCorners, 0)