import flatbuffers
from flatbuffers.compat import import_numpy
def CallOnceOptionsAddInitSubgraphIndex(builder, initSubgraphIndex):
    builder.PrependInt32Slot(0, initSubgraphIndex, 0)