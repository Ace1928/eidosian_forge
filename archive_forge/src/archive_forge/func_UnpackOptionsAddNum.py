import flatbuffers
from flatbuffers.compat import import_numpy
def UnpackOptionsAddNum(builder, num):
    builder.PrependInt32Slot(0, num, 0)