import flatbuffers
from flatbuffers.compat import import_numpy
def VarHandleOptionsAddContainer(builder, container):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(container), 0)