import flatbuffers
from flatbuffers.compat import import_numpy
def HashtableOptionsAddKeyDtype(builder, keyDtype):
    builder.PrependInt8Slot(1, keyDtype, 0)