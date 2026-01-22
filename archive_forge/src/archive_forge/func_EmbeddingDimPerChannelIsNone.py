import flatbuffers
from flatbuffers.compat import import_numpy
def EmbeddingDimPerChannelIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
    return o == 0