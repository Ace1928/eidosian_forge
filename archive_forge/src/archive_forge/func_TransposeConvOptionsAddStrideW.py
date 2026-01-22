import flatbuffers
from flatbuffers.compat import import_numpy
def TransposeConvOptionsAddStrideW(builder, strideW):
    builder.PrependInt32Slot(1, strideW, 0)