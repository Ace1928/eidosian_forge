import flatbuffers
from flatbuffers.compat import import_numpy
def ModelAddOperatorCodes(builder, operatorCodes):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(operatorCodes), 0)