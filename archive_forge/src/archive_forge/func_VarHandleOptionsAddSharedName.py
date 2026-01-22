import flatbuffers
from flatbuffers.compat import import_numpy
def VarHandleOptionsAddSharedName(builder, sharedName):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(sharedName), 0)