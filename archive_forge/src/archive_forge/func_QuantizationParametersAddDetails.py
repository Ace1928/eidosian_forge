import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationParametersAddDetails(builder, details):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(details), 0)