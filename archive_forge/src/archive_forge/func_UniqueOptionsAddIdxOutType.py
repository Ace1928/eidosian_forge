import flatbuffers
from flatbuffers.compat import import_numpy
def UniqueOptionsAddIdxOutType(builder, idxOutType):
    builder.PrependInt8Slot(0, idxOutType, 2)