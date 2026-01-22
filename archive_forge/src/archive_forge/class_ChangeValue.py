from sys import version_info as _swig_python_version_info
import weakref
class ChangeValue(IntVarLocalSearchOperator):
    """
    Defines operators which change the value of variables;
    each neighbor corresponds to *one* modified variable.
    Sub-classes have to define ModifyValue which determines what the new
    variable value is going to be (given the current value and the variable).
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, vars):
        if self.__class__ == ChangeValue:
            _self = None
        else:
            _self = self
        _pywrapcp.ChangeValue_swiginit(self, _pywrapcp.new_ChangeValue(_self, vars))
    __swig_destroy__ = _pywrapcp.delete_ChangeValue

    def ModifyValue(self, index, value):
        return _pywrapcp.ChangeValue_ModifyValue(self, index, value)

    def OneNeighbor(self):
        """ This method should not be overridden. Override ModifyValue() instead."""
        return _pywrapcp.ChangeValue_OneNeighbor(self)

    def __disown__(self):
        self.this.disown()
        _pywrapcp.disown_ChangeValue(self)
        return weakref.proxy(self)