import flatbuffers
from flatbuffers.compat import import_numpy
def CustomOptions(self, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
    if o != 0:
        a = self._tab.Vector(o)
        return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
    return 0