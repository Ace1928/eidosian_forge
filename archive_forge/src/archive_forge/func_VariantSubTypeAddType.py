import flatbuffers
from flatbuffers.compat import import_numpy
def VariantSubTypeAddType(builder, type):
    builder.PrependInt8Slot(1, type, 0)