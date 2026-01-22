import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationParametersAddMin(builder, min):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(min), 0)