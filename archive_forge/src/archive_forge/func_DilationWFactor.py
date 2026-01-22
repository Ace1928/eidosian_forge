import flatbuffers
from flatbuffers.compat import import_numpy
def DilationWFactor(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
    if o != 0:
        return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
    return 1