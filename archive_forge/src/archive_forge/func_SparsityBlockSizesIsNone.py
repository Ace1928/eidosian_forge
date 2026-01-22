import flatbuffers
from flatbuffers.compat import import_numpy
def SparsityBlockSizesIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
    return o == 0