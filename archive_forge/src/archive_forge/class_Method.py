import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
class Method(Attribute):
    """Method interfaces

    The idea here is that you have objects that describe methods.
    This provides an opportunity for rich meta-data.
    """
    positional = required = ()
    _optional = varargs = kwargs = None

    def _get_optional(self):
        if self._optional is None:
            return {}
        return self._optional

    def _set_optional(self, opt):
        self._optional = opt

    def _del_optional(self):
        self._optional = None
    optional = property(_get_optional, _set_optional, _del_optional)

    def __call__(self, *args, **kw):
        raise BrokenImplementation(self.interface, self.__name__)

    def getSignatureInfo(self):
        return {'positional': self.positional, 'required': self.required, 'optional': self.optional, 'varargs': self.varargs, 'kwargs': self.kwargs}

    def getSignatureString(self):
        sig = []
        for v in self.positional:
            sig.append(v)
            if v in self.optional.keys():
                sig[-1] += '=' + repr(self.optional[v])
        if self.varargs:
            sig.append('*' + self.varargs)
        if self.kwargs:
            sig.append('**' + self.kwargs)
        return '(%s)' % ', '.join(sig)
    _get_str_info = getSignatureString