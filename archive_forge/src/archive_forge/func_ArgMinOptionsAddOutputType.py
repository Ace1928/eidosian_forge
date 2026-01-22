import flatbuffers
from flatbuffers.compat import import_numpy
def ArgMinOptionsAddOutputType(builder, outputType):
    builder.PrependInt8Slot(0, outputType, 0)