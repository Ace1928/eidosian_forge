import flatbuffers
from flatbuffers.compat import import_numpy
def ConversionOptionsAddEnableSelectTfOps(builder, enableSelectTfOps):
    builder.PrependBoolSlot(2, enableSelectTfOps, 0)