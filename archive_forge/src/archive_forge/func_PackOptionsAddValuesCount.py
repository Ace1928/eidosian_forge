import flatbuffers
from flatbuffers.compat import import_numpy
def PackOptionsAddValuesCount(builder, valuesCount):
    builder.PrependInt32Slot(0, valuesCount, 0)