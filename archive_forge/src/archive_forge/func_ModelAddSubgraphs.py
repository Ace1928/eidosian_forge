import flatbuffers
from flatbuffers.compat import import_numpy
def ModelAddSubgraphs(builder, subgraphs):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(subgraphs), 0)