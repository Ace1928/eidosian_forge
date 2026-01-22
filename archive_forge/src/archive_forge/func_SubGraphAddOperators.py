import flatbuffers
from flatbuffers.compat import import_numpy
def SubGraphAddOperators(builder, operators):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(operators), 0)