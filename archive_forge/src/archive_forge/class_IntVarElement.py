from sys import version_info as _swig_python_version_info
import weakref
class IntVarElement(AssignmentElement):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr

    def Var(self):
        return _pywrapcp.IntVarElement_Var(self)

    def Min(self):
        return _pywrapcp.IntVarElement_Min(self)

    def SetMin(self, m):
        return _pywrapcp.IntVarElement_SetMin(self, m)

    def Max(self):
        return _pywrapcp.IntVarElement_Max(self)

    def SetMax(self, m):
        return _pywrapcp.IntVarElement_SetMax(self, m)

    def Value(self):
        return _pywrapcp.IntVarElement_Value(self)

    def Bound(self):
        return _pywrapcp.IntVarElement_Bound(self)

    def SetRange(self, l, u):
        return _pywrapcp.IntVarElement_SetRange(self, l, u)

    def SetValue(self, v):
        return _pywrapcp.IntVarElement_SetValue(self, v)

    def __eq__(self, element):
        return _pywrapcp.IntVarElement___eq__(self, element)

    def __ne__(self, element):
        return _pywrapcp.IntVarElement___ne__(self, element)
    __swig_destroy__ = _pywrapcp.delete_IntVarElement