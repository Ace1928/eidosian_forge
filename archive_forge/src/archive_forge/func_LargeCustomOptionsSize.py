import flatbuffers
from flatbuffers.compat import import_numpy
def LargeCustomOptionsSize(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
    if o != 0:
        return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
    return 0