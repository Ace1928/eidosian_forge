import flatbuffers
from flatbuffers.compat import import_numpy
def SparsityParametersAddDimMetadata(builder, dimMetadata):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dimMetadata), 0)