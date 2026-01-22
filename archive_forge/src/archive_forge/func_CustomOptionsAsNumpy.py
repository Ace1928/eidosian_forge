import flatbuffers
from flatbuffers.compat import import_numpy
def CustomOptionsAsNumpy(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
    if o != 0:
        return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
    return 0