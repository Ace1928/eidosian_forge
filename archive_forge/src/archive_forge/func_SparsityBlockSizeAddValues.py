import flatbuffers
from flatbuffers.compat import import_numpy
def SparsityBlockSizeAddValues(builder, values):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0)