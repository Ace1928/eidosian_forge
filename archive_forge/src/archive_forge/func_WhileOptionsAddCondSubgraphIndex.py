import flatbuffers
from flatbuffers.compat import import_numpy
def WhileOptionsAddCondSubgraphIndex(builder, condSubgraphIndex):
    builder.PrependInt32Slot(0, condSubgraphIndex, 0)