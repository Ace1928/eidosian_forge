import flatbuffers
from flatbuffers.compat import import_numpy
def ShapeOptionsAddOutType(builder, outType):
    builder.PrependInt8Slot(0, outType, 0)