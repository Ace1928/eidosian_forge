import flatbuffers
from flatbuffers.compat import import_numpy
def SparsityParametersAddBlockMap(builder, blockMap):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(blockMap), 0)