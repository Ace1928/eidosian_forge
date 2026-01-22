import flatbuffers
from flatbuffers.compat import import_numpy
def UnpackOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(1, axis, 0)