import flatbuffers
from flatbuffers.compat import import_numpy
def Conv3DOptionsAddStrideD(builder, strideD):
    builder.PrependInt32Slot(1, strideD, 0)