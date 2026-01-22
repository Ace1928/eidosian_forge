from sys import version_info as _swig_python_version_info
import weakref
class TypeIncompatibilityChecker(TypeRegulationsChecker):
    """ Checker for type incompatibilities."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, model, check_hard_incompatibilities):
        _pywrapcp.TypeIncompatibilityChecker_swiginit(self, _pywrapcp.new_TypeIncompatibilityChecker(model, check_hard_incompatibilities))
    __swig_destroy__ = _pywrapcp.delete_TypeIncompatibilityChecker