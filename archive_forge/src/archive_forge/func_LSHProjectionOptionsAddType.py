import flatbuffers
from flatbuffers.compat import import_numpy
def LSHProjectionOptionsAddType(builder, type):
    builder.PrependInt8Slot(0, type, 0)