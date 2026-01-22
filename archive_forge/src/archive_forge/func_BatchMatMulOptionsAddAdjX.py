import flatbuffers
from flatbuffers.compat import import_numpy
def BatchMatMulOptionsAddAdjX(builder, adjX):
    builder.PrependBoolSlot(0, adjX, 0)