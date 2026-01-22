import flatbuffers
from flatbuffers.compat import import_numpy
def OneHotOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)