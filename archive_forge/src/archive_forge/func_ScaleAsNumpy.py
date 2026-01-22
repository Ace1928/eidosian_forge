import flatbuffers
from flatbuffers.compat import import_numpy
def ScaleAsNumpy(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
    if o != 0:
        return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
    return 0