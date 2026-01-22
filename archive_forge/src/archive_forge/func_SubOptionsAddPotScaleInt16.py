import flatbuffers
from flatbuffers.compat import import_numpy
def SubOptionsAddPotScaleInt16(builder, potScaleInt16):
    builder.PrependBoolSlot(1, potScaleInt16, 1)