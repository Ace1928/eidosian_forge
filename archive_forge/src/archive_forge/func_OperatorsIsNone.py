import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorsIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
    return o == 0