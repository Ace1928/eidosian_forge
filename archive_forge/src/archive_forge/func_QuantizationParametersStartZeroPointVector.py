import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationParametersStartZeroPointVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)