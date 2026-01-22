import flatbuffers
from flatbuffers.compat import import_numpy
def DepthToSpaceOptionsAddBlockSize(builder, blockSize):
    builder.PrependInt32Slot(0, blockSize, 0)