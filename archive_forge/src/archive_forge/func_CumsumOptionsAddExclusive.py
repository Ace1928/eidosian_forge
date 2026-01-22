import flatbuffers
from flatbuffers.compat import import_numpy
def CumsumOptionsAddExclusive(builder, exclusive):
    builder.PrependBoolSlot(0, exclusive, 0)