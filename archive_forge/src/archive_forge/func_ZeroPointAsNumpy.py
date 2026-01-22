import flatbuffers
from flatbuffers.compat import import_numpy
def ZeroPointAsNumpy(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
    if o != 0:
        return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
    return 0