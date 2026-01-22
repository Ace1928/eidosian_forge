import flatbuffers
from flatbuffers.compat import import_numpy
def SpaceToDepthOptionsAddBlockSize(builder, blockSize):
    builder.PrependInt32Slot(0, blockSize, 0)