import flatbuffers
from flatbuffers.compat import import_numpy
def DimensionMetadataAddArraySegmentsType(builder, arraySegmentsType):
    builder.PrependUint8Slot(2, arraySegmentsType, 0)