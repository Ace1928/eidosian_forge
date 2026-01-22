import flatbuffers
from flatbuffers.compat import import_numpy
def Sparsity(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
    if o != 0:
        x = self._tab.Indirect(o + self._tab.Pos)
        obj = SparsityParameters()
        obj.Init(self._tab.Bytes, x)
        return obj
    return None