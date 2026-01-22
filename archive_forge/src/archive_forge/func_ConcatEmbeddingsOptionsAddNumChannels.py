import flatbuffers
from flatbuffers.compat import import_numpy
def ConcatEmbeddingsOptionsAddNumChannels(builder, numChannels):
    builder.PrependInt32Slot(0, numChannels, 0)