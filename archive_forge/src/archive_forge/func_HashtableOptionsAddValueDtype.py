import flatbuffers
from flatbuffers.compat import import_numpy
def HashtableOptionsAddValueDtype(builder, valueDtype):
    builder.PrependInt8Slot(2, valueDtype, 0)