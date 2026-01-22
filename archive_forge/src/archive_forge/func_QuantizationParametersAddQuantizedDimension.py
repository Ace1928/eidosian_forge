import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationParametersAddQuantizedDimension(builder, quantizedDimension):
    builder.PrependInt32Slot(6, quantizedDimension, 0)