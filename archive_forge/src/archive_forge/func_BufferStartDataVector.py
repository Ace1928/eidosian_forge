import flatbuffers
from flatbuffers.compat import import_numpy
def BufferStartDataVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)