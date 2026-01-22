from sys import version_info as _swig_python_version_info
import weakref
class IntVarContainer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr

    def Contains(self, var):
        return _pywrapcp.IntVarContainer_Contains(self, var)

    def Element(self, index):
        return _pywrapcp.IntVarContainer_Element(self, index)

    def Size(self):
        return _pywrapcp.IntVarContainer_Size(self)

    def Store(self):
        return _pywrapcp.IntVarContainer_Store(self)

    def Restore(self):
        return _pywrapcp.IntVarContainer_Restore(self)

    def __eq__(self, container):
        """
        Returns true if this and 'container' both represent the same V* -> E map.
        Runs in linear time; requires that the == operator on the type E is well
        defined.
        """
        return _pywrapcp.IntVarContainer___eq__(self, container)

    def __ne__(self, container):
        return _pywrapcp.IntVarContainer___ne__(self, container)
    __swig_destroy__ = _pywrapcp.delete_IntVarContainer