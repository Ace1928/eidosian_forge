import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorCodesLength(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
    if o != 0:
        return self._tab.VectorLen(o)
    return 0