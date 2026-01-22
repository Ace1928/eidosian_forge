import flatbuffers
from flatbuffers.compat import import_numpy
def DimensionMetadataAddFormat(builder, format):
    builder.PrependInt8Slot(0, format, 0)