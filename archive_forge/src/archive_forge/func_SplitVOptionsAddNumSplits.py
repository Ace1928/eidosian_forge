import flatbuffers
from flatbuffers.compat import import_numpy
def SplitVOptionsAddNumSplits(builder, numSplits):
    builder.PrependInt32Slot(0, numSplits, 0)