import flatbuffers
from flatbuffers.compat import import_numpy
def RandomOptionsAddSeed(builder, seed):
    builder.PrependInt64Slot(0, seed, 0)