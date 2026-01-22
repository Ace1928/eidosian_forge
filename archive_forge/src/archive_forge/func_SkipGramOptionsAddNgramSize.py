import flatbuffers
from flatbuffers.compat import import_numpy
def SkipGramOptionsAddNgramSize(builder, ngramSize):
    builder.PrependInt32Slot(0, ngramSize, 0)