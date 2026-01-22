import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorStartCustomOptionsVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)